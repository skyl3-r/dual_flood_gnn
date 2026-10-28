import torch
from torch import Tensor
from torch.nn import ModuleList
from torch_geometric.utils import scatter

from utils.model_utils import make_mlp
from .base_model import BaseModel
from .dual_flood_gnn import NodeEdgeConv


class HierarchicalDUALFloodGNN(BaseModel):
    """Multilevel METIS hierarchy with latent-only coarse communication."""

    requires_hierarchy = True

    def __init__(self, hidden_features=64, num_layers=3, num_hierarchy_levels=1,
                 activation='relu', residual=True, mlp_layers=2, encoder_layers=3,
                 encoder_activation='relu', decoder_layers=3,
                 decoder_activation='relu', **base_model_kwargs):
        super().__init__(**base_model_kwargs)
        if num_hierarchy_levels < 1:
            raise ValueError('num_hierarchy_levels must be at least 1')

        self.hidden_features = hidden_features
        self.num_hierarchy_levels = num_hierarchy_levels
        self.num_levels = num_hierarchy_levels + 1

        self.node_encoder = make_mlp(self.input_node_features, hidden_features,
                                     hidden_features * 2, encoder_layers,
                                     encoder_activation, bias=False, device=self.device)
        self.edge_encoder = make_mlp(self.input_edge_features, hidden_features,
                                     hidden_features, encoder_layers,
                                     encoder_activation, bias=False, device=self.device)

        def make_processor():
            return ModuleList([
                NodeEdgeConv(hidden_features, hidden_features,
                             hidden_features, hidden_features,
                             hidden_features, mlp_layers, activation,
                             residual, False, self.device)
                for _ in range(num_layers)
            ])

        self.level_processors = ModuleList([
            make_processor() for _ in range(self.num_levels)
        ])
        self.level_refinements = ModuleList([
            NodeEdgeConv(hidden_features, hidden_features,
                         hidden_features, hidden_features,
                         hidden_features, mlp_layers, activation,
                         residual, False, self.device)
            for _ in range(self.num_levels)
        ])

        # One static edge encoder, cross-level encoder, upsampler, and fusion
        # module per child-to-parent transition.
        self.level_edge_encoders = ModuleList([
            make_mlp(4, hidden_features, hidden_features,
                     max(1, encoder_layers), encoder_activation,
                     bias=False, device=self.device)
            for _ in range(num_hierarchy_levels)
        ])
        self.cross_encoders = ModuleList([
            make_mlp(2, hidden_features, hidden_features,
                     max(1, encoder_layers), encoder_activation,
                     bias=False, device=self.device)
            for _ in range(num_hierarchy_levels)
        ])
        self.up_mlps = ModuleList([
            make_mlp(hidden_features * 2, hidden_features, hidden_features * 2,
                     mlp_layers, activation, bias=False, device=self.device)
            for _ in range(num_hierarchy_levels)
        ])
        self.fusion_mlps = ModuleList([
            make_mlp(hidden_features * 2, hidden_features, hidden_features * 2,
                     mlp_layers, activation, bias=False, device=self.device)
            for _ in range(num_hierarchy_levels)
        ])

        self.node_decoder = make_mlp(hidden_features, self.output_node_features,
                                     hidden_features * 2, decoder_layers,
                                     decoder_activation, bias=False, device=self.device)
        self.edge_decoder = make_mlp(hidden_features, self.output_edge_features,
                                     hidden_features * 2, decoder_layers,
                                     decoder_activation, bias=False, device=self.device)

    @staticmethod
    def _hierarchy_value(hierarchy, level, name):
        value = getattr(hierarchy, f'hierarchy_{level}_{name}', None)
        if value is None:
            raise ValueError(
                f'Missing hierarchy_{level}_{name}; hierarchy has insufficient levels '
                'for this model configuration.'
            )
        return value

    def _process_level(self, level, x, edge_index, edge_attr):
        for layer in self.level_processors[level]:
            x, edge_attr = layer(x, edge_index, edge_attr)
        return x, edge_attr

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, hierarchy=None):
        if hierarchy is None:
            raise ValueError('Pass the PyG Data object as hierarchy=... when using HierarchicalDUALFloodGNN')

        hierarchy_levels = int(hierarchy.hierarchy_num_levels.item())
        if hierarchy_levels != self.num_hierarchy_levels:
            raise ValueError(
                f'Model expects {self.num_hierarchy_levels} hierarchy levels, '
                f'but data contains {hierarchy_levels}.'
            )

        x_levels = [self.node_encoder(x)]
        edge_attrs = [self.edge_encoder(edge_attr)]
        edge_indices = [edge_index]
        skips = [None] * self.num_levels

        # Encoder path: process a resolution, save its skip, then mean-pool
        # node states to initialize the next coarser resolution.
        for level in range(self.num_levels):
            x_levels[level], edge_attrs[level] = self._process_level(
                level, x_levels[level], edge_indices[level], edge_attrs[level]
            )
            if level < self.num_hierarchy_levels:
                skips[level] = x_levels[level]
                cluster = self._hierarchy_value(hierarchy, level, 'cluster').long()
                num_nodes = int(self._hierarchy_value(hierarchy, level, 'num_nodes').item())
                x_levels.append(scatter(x_levels[level], cluster, dim=0,
                                        dim_size=num_nodes, reduce='mean'))
                edge_indices.append(self._hierarchy_value(hierarchy, level, 'edge_index').long())
                static_edge_attr = self._hierarchy_value(hierarchy, level, 'edge_attr')
                edge_attrs.append(self.level_edge_encoders[level](static_edge_attr))

        # Decoder path: learned parent-to-child transfer followed by explicit
        # U-Net skip fusion at every resolution.
        for level in reversed(range(self.num_hierarchy_levels)):
            cross = self._hierarchy_value(hierarchy, level, 'cross_edge_index').long()
            child_ids, parent_ids = cross[0], cross[1]
            cross_attr = self.cross_encoders[level](
                self._hierarchy_value(hierarchy, level, 'cross_edge_attr')
            )
            up_messages = self.up_mlps[level](torch.cat([
                x_levels[level + 1][parent_ids], cross_attr
            ], dim=-1))
            upsampled = scatter(up_messages, child_ids, dim=0,
                                dim_size=x_levels[level].size(0), reduce='sum')
            x_levels[level] = self.fusion_mlps[level](torch.cat([
                skips[level], upsampled
            ], dim=-1))
            x_levels[level], edge_attrs[level] = self.level_refinements[level](
                x_levels[level], edge_indices[level], edge_attrs[level]
            )

        return self.node_decoder(x_levels[0]), self.edge_decoder(edge_attrs[0])
