"""Static recursive METIS hierarchy construction for hierarchical GNNs."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def metis_partition(edge_index: Tensor, num_nodes: int, num_clusters: int) -> Tensor:
    """Return one contiguous METIS cluster id per node."""
    if num_clusters >= num_nodes:
        return torch.arange(num_nodes, dtype=torch.long)
    if num_clusters < 1:
        raise ValueError("num_clusters must be positive")
    try:
        import pymetis
    except ImportError as exc:
        raise ImportError("HierarchicalDUALFloodGNN requires pymetis") from exc

    undirected = to_undirected(edge_index, num_nodes=num_nodes).cpu().t().tolist()
    adjacency = [[] for _ in range(num_nodes)]
    for u, v in undirected:
        if u != v:
            adjacency[u].append(v)
    _, parts = pymetis.part_graph(num_clusters, adjacency=adjacency)
    parts = torch.tensor(parts, dtype=torch.long)
    # PyMetis normally returns 0..K-1, but relabel defensively so every
    # downstream scatter has a dense parent index space.
    _, parts = torch.unique(parts, sorted=True, return_inverse=True)
    return parts


def _build_transition(
    edge_index: Tensor,
    static_nodes: np.ndarray,
    static_edges: np.ndarray,
    positions: Optional[np.ndarray],
    ratio: float,
) -> tuple[dict[str, Tensor], np.ndarray, np.ndarray, np.ndarray]:
    """Build one child-to-parent transition and its parent graph metadata."""
    if not 0 < ratio <= 1:
        raise ValueError(f"hierarchy ratio must be in (0, 1], got {ratio}")

    n = int(static_nodes.shape[0])
    requested_k = min(n, max(1, math.ceil(ratio * n)))
    part = metis_partition(edge_index, n, requested_k)
    k = int(part.max().item()) + 1

    area = torch.as_tensor(static_nodes[:, 0], dtype=torch.float32).abs().clamp_min(1e-8)
    cluster_area = torch.zeros(k, dtype=torch.float32).scatter_add_(0, part, area)
    if positions is None:
        positions = np.arange(n, dtype=np.float32)[:, None]
        positions = np.pad(positions, ((0, 0), (0, 1)))
    pos = torch.as_tensor(positions[:, :2], dtype=torch.float32)

    centroids = torch.zeros((k, 2), dtype=torch.float32)
    centroids.index_add_(0, part, pos * area[:, None])
    centroids = centroids / cluster_area[:, None]

    elevation = torch.as_tensor(static_nodes[:, 2], dtype=torch.float32)
    mean_elevation = torch.zeros(k, dtype=torch.float32).scatter_add_(0, part, elevation * area)
    mean_elevation = mean_elevation / cluster_area

    # Area is summed; remaining static features are area-weighted means.
    parent_nodes = np.zeros((k, static_nodes.shape[1]), dtype=np.float32)
    parent_nodes[:, 0] = cluster_area.numpy()
    for feature_idx in range(1, static_nodes.shape[1]):
        values = torch.as_tensor(static_nodes[:, feature_idx], dtype=torch.float32)
        weighted = torch.zeros(k, dtype=torch.float32).scatter_add_(0, part, values * area)
        parent_nodes[:, feature_idx] = (weighted / cluster_area).numpy()

    interfaces: dict[tuple[int, int], list[float]] = {}
    for edge_id, (u, v) in enumerate(edge_index.t().tolist()):
        cu, cv = int(part[u]), int(part[v])
        if cu != cv:
            length = float(static_edges[edge_id, 0]) if static_edges.shape[1] else 1.0
            interfaces.setdefault((cu, cv), []).append(length)

    coarse_edges, coarse_features = [], []
    for (cu, cv), lengths in interfaces.items():
        distance = torch.linalg.vector_norm(centroids[cv] - centroids[cu]).item()
        slope = (mean_elevation[cv] - mean_elevation[cu]).item() / (distance + 1e-8)
        coarse_edges.append((cu, cv))
        coarse_features.append([distance, float(sum(lengths)), float(len(lengths)), slope])

    if coarse_edges:
        parent_edge_index = torch.tensor(coarse_edges, dtype=torch.long).t().contiguous()
        parent_edge_attr = torch.tensor(coarse_features, dtype=torch.float32)
    else:
        parent_edge_index = torch.empty((2, 0), dtype=torch.long)
        parent_edge_attr = torch.empty((0, 4), dtype=torch.float32)

    boundary = torch.zeros(n, dtype=torch.float32)
    for u, v in edge_index.t().tolist():
        if part[u] != part[v]:
            boundary[u] = boundary[v] = 1.0

    transition = {
        "cluster": part,
        "edge_index": parent_edge_index,
        "edge_attr": parent_edge_attr,
        "cross_edge_index": torch.stack([torch.arange(n), part], dim=0),
        "cross_edge_attr": torch.stack([area / cluster_area[part], boundary], dim=-1),
        "num_nodes": torch.tensor(k, dtype=torch.long),
    }
    return transition, parent_nodes, parent_edge_index, centroids.numpy()


def build_hierarchy(
    edge_index: Tensor,
    static_nodes: np.ndarray,
    static_edges: np.ndarray,
    positions: Optional[np.ndarray],
    ratios: Sequence[float] | float,
) -> dict[str, Tensor]:
    """Build a recursive hierarchy and return flattened, cacheable tensors."""
    if isinstance(ratios, (float, int)):
        ratios = [float(ratios)]
    ratios = [float(ratio) for ratio in ratios]
    if not ratios:
        raise ValueError("At least one hierarchy ratio is required")

    result: dict[str, Tensor] = {
        "hierarchy_num_levels": torch.tensor(len(ratios), dtype=torch.long)
    }
    current_edge_index = edge_index.long()
    current_nodes = static_nodes
    current_edges = static_edges
    current_positions = positions

    for level, ratio in enumerate(ratios):
        transition, current_nodes, current_edge_index, current_positions = _build_transition(
            current_edge_index, current_nodes, current_edges, current_positions, ratio
        )
        current_edges = transition["edge_attr"].numpy()
        for key, value in transition.items():
            result[f"hierarchy_{level}_{key}"] = value

    return result


def attach_hierarchy(data: Data, static_values: dict, ratios=None) -> Data:
    """Attach indexed hierarchy arrays to a timestep Data object."""
    if "hierarchy_num_levels" not in static_values:
        if "cluster" in static_values:
            static_values = {
                "hierarchy_num_levels": torch.tensor(1, dtype=torch.long),
                **{f"hierarchy_0_{key}": value for key, value in static_values.items()
                   if key in ("cluster", "coarse_edge_index", "coarse_edge_attr",
                              "cross_edge_index", "cross_edge_attr")},
                "hierarchy_0_num_nodes": static_values["num_supernodes"],
            }
        else:
            return data

    num_levels = int(static_values["hierarchy_num_levels"])
    data.hierarchy_num_levels = static_values["hierarchy_num_levels"]
    for level in range(num_levels):
        for key in ("cluster", "edge_index", "edge_attr", "cross_edge_index",
                    "cross_edge_attr", "num_nodes"):
            name = f"hierarchy_{level}_{key}"
            if name not in static_values:
                raise KeyError(f"Missing hierarchy tensor: {name}")
            setattr(data, name, static_values[name])
    return data
