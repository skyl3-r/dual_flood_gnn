"""Static one-level METIS hierarchy construction for mSWE-GNN graphs."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def metis_partition(edge_index: Tensor, num_nodes: int, num_clusters: int) -> Tensor:
    """Return one METIS cluster id per node.

    PyMetis is deliberately imported here (rather than at module import time) so
    the rest of the dataset remains usable when hierarchy support is disabled.
    """
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
    return torch.tensor(parts, dtype=torch.long)


def build_hierarchy(
    edge_index: Tensor,
    static_nodes: np.ndarray,
    static_edges: np.ndarray,
    positions: Optional[np.ndarray],
    ratio: float,
) -> dict[str, Tensor]:
    """Build and return all static tensors needed by the hierarchical model.

    ``static_nodes`` follows FloodEventDataset's order: area, roughness,
    elevation, ...; ``static_edges`` starts with face/interface length.
    """
    if not 0 < ratio <= 1:
        raise ValueError(f"hierarchy ratio must be in (0, 1], got {ratio}")
    n = int(static_nodes.shape[0])
    k = min(n, max(1, math.ceil(ratio * n)))
    part = metis_partition(edge_index, n, k)

    area = torch.as_tensor(static_nodes[:, 0], dtype=torch.float32)
    elevation = torch.as_tensor(static_nodes[:, 2], dtype=torch.float32)
    # Area is positive in the raw dataset. The clamp also keeps malformed or
    # normalized inputs from producing NaNs in cached metadata.
    area = area.abs().clamp_min(1e-8)
    cluster_area = torch.zeros(k, dtype=torch.float32).scatter_add_(0, part, area)
    centroids = torch.zeros((k, 2), dtype=torch.float32)
    if positions is None:
        positions = np.arange(n, dtype=np.float32)[:, None]
        positions = np.pad(positions, ((0, 0), (0, 1)))
    pos = torch.as_tensor(positions[:, :2], dtype=torch.float32)
    centroids.index_add_(0, part, pos * area[:, None])
    centroids = centroids / cluster_area[:, None]
    mean_elevation = torch.zeros(k, dtype=torch.float32).scatter_add_(0, part, elevation * area) / cluster_area

    # Keep one directed edge per direction, and aggregate the fine interface.
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
        coarse_edge_index = torch.tensor(coarse_edges, dtype=torch.long).t().contiguous()
        coarse_edge_attr = torch.tensor(coarse_features, dtype=torch.float32)
    else:
        coarse_edge_index = torch.empty((2, 0), dtype=torch.long)
        coarse_edge_attr = torch.empty((0, 4), dtype=torch.float32)

    boundary = torch.zeros(n, dtype=torch.float32)
    for u, v in edge_index.t().tolist():
        if part[u] != part[v]:
            boundary[u] = boundary[v] = 1.0
    cross_edge_index = torch.stack([torch.arange(n), part], dim=0)
    cross_attr = torch.stack([area / cluster_area[part], boundary], dim=-1)
    return {
        "cluster": part,
        "coarse_edge_index": coarse_edge_index,
        "coarse_edge_attr": coarse_edge_attr,
        "cross_edge_index": cross_edge_index,
        "cross_edge_attr": cross_attr,
        "num_supernodes": torch.tensor(k, dtype=torch.long),
    }


def attach_hierarchy(data: Data, static_values: dict, ratio: float) -> Data:
    """Attach cached hierarchy arrays to a timestep Data object."""
    for key in ("cluster", "coarse_edge_index", "coarse_edge_attr", "cross_edge_index", "cross_edge_attr"):
        setattr(data, key, static_values[key])
    data.num_supernodes = static_values["num_supernodes"]
    return data
