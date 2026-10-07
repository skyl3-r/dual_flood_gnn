import torch
from torch import Tensor
from torch.nn import ModuleList
from torch_geometric.utils import scatter

from utils.model_utils import make_mlp
from .base_model import BaseModel
from .dual_flood_gnn import NodeEdgeConv


class HierarchicalDUALFloodGNN(BaseModel):
    """One-level METIS hierarchy with latent-only coarse communication."""
    requires_hierarchy = True

    def __init__(self, hidden_features=64, num_layers=3, activation='relu',
                 residual=True, mlp_layers=2, encoder_layers=3,
                 encoder_activation='relu', decoder_layers=3,
                 decoder_activation='relu', **base_model_kwargs):
        super().__init__(**base_model_kwargs)
        self.hidden_features = hidden_features
        self.node_encoder = make_mlp(self.input_node_features, hidden_features,
                                     hidden_features * 2, encoder_layers,
                                     encoder_activation, bias=False, device=self.device)
        self.edge_encoder = make_mlp(self.input_edge_features, hidden_features,
                                     hidden_features, encoder_layers,
                                     encoder_activation, bias=False, device=self.device)
        self.fine_layers = ModuleList([NodeEdgeConv(hidden_features, hidden_features,
                                                     hidden_features, hidden_features,
                                                     hidden_features, mlp_layers, activation,
                                                     residual, False, self.device)
                                       for _ in range(num_layers)])
        self.coarse_edge_encoder = make_mlp(4, hidden_features, hidden_features,
                                            max(1, encoder_layers), encoder_activation,
                                            bias=False, device=self.device)
        self.coarse_layers = ModuleList([NodeEdgeConv(hidden_features, hidden_features,
                                                       hidden_features, hidden_features,
                                                       hidden_features, mlp_layers, activation,
                                                       residual, False, self.device)
                                         for _ in range(num_layers)])
        self.cross_encoder = make_mlp(2, hidden_features, hidden_features,
                                      max(1, encoder_layers), encoder_activation,
                                      bias=False, device=self.device)
        self.up_mlp = make_mlp(hidden_features * 3, hidden_features, hidden_features * 2,
                               mlp_layers, activation, bias=False, device=self.device)
        self.node_decoder = make_mlp(hidden_features, self.output_node_features,
                                     hidden_features * 2, decoder_layers,
                                     decoder_activation, bias=False, device=self.device)
        self.edge_decoder = make_mlp(hidden_features, self.output_edge_features,
                                     hidden_features * 2, decoder_layers,
                                     decoder_activation, bias=False, device=self.device)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, hierarchy=None):
        if hierarchy is None:
            hierarchy = self._hierarchy_from_data()
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        for layer in self.fine_layers:
            x, edge_attr = layer(x, edge_index, edge_attr)

        cluster = hierarchy.cluster.long()
        num_supernodes = int(hierarchy.num_supernodes.item())
        coarse_x = scatter(x, cluster, dim=0, dim_size=num_supernodes, reduce='mean')
        coarse_edge_attr = self.coarse_edge_encoder(hierarchy.coarse_edge_attr)
        for layer in self.coarse_layers:
            coarse_x, coarse_edge_attr = layer(coarse_x, hierarchy.coarse_edge_index, coarse_edge_attr)

        cross = hierarchy.cross_edge_index
        fine_ids, coarse_ids = cross[0], cross[1]
        cross_attr = self.cross_encoder(hierarchy.cross_edge_attr)
        up_messages = self.up_mlp(torch.cat([coarse_x[coarse_ids], x[fine_ids], cross_attr], dim=-1))
        x = x + scatter(up_messages, fine_ids, dim=0, dim_size=x.size(0), reduce='sum')
        return self.node_decoder(x), self.edge_decoder(edge_attr)

    def _hierarchy_from_data(self):
        raise ValueError('Pass the PyG Data object as hierarchy=... when using HierarchicalDUALFloodGNN')
