#!/usr/bin/env python3
"""Test the original mSWE-GNN boundary pipeline with all NetCDF edges.

This does not train a model. It verifies that an all-edge export can enter the
original boundary-condition sequence, have wall edges removed, and retain the
single inflow boundary edge after reapplication.

Example:
    python data_mswegnn/test_original_boundary_pipeline.py \
        --map-nc data_mswegnn/datasets/raw/Simulations/output_1_map.nc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# Allow execution as `python data_mswegnn/test_original_boundary_pipeline.py`
# from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.boundary_condition import BoundaryCondition


def normalise_connectivity(values: np.ndarray, expected_rows: int) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D connectivity, got {values.shape}")
    if values.shape[0] == expected_rows:
        return values
    if values.shape[1] == expected_rows:
        return values.T
    raise ValueError(f"No connectivity dimension matches {expected_rows}: {values.shape}")


def zero_based(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, -1, dtype=np.int64)
    valid = np.isfinite(values) & (values > 0)
    result[valid] = values[valid].astype(np.int64) - 1
    return result


def run_test(map_nc: Path) -> None:
    with xr.open_dataset(map_nc) as ds:
        n_faces = ds["mesh2d_face_x"].size
        n_edges = ds["mesh2d_edge_x"].size

        edge_faces_raw = normalise_connectivity(
            ds["mesh2d_edge_faces"].values, n_edges
        )
        edge_faces = zero_based(edge_faces_raw)
        edge_types = ds["mesh2d_edge_type"].values.astype(np.int64).reshape(-1)

        if edge_faces.shape != (n_edges, 2):
            raise AssertionError(f"Unexpected edge_faces shape: {edge_faces.shape}")
        if edge_types.shape != (n_edges,):
            raise AssertionError("edge_type does not match the number of edges")

        boundary = np.any(edge_faces < 0, axis=1)
        type2_edges = np.flatnonzero((edge_types == 2) & boundary)
        type3_edges = np.flatnonzero((edge_types == 3) & boundary)
        if len(type2_edges) != 1:
            raise AssertionError(
                f"Expected exactly one type-2 inflow edge; found {len(type2_edges)}"
            )

        # Append one ghost node for every type-2/type-3 boundary edge. This is
        # the all-edge representation being tested.
        ghost_edge_ids = np.r_[type2_edges, type3_edges]
        n_ghost = len(ghost_edge_ids)
        ghost_nodes = np.arange(n_faces, n_faces + n_ghost, dtype=np.int64)
        inflow_ghost = ghost_nodes[0]

        edge_index = edge_faces.T.copy()
        for offset, edge_id in enumerate(ghost_edge_ids):
            valid_faces = edge_faces[edge_id][edge_faces[edge_id] >= 0]
            if len(valid_faces) != 1:
                raise AssertionError(f"Boundary edge {edge_id} has invalid faces")
            if edge_id in type2_edges:
                # The original boundary pipeline expects inflow to already be
                # oriented ghost -> real; otherwise it reverses and negates
                # the dynamic flow during create().
                edge_index[:, edge_id] = [ghost_nodes[offset], valid_faces[0]]
            else:
                edge_index[:, edge_id] = [valid_faces[0], ghost_nodes[offset]]

        # Read q1 only to verify the initial all-edge alignment. Boundary
        # processing itself operates on feature tensors, not raw q1 arrays.
        q1 = np.asarray(ds["mesh2d_q1"].values)
        if q1.ndim != 2:
            raise AssertionError(f"mesh2d_q1 must be 2D, got {q1.shape}")
        if n_edges not in q1.shape:
            raise AssertionError(f"mesh2d_q1 has no edge dimension: {q1.shape}")
        q1 = q1 if q1.shape[1] == n_edges else q1.T

    # Construct the same array shapes used by FloodEventDataset.process().
    node_types = np.r_[
        np.ones(n_faces, dtype=np.int64),
        np.full(n_ghost, 3, dtype=np.int64),
    ]
    node_types[inflow_ghost] = 2

    static_nodes = np.zeros((n_faces + n_ghost, 1), dtype=np.float32)
    dynamic_nodes = np.zeros((q1.shape[0], n_faces + n_ghost, 1), dtype=np.float32)
    static_edges = np.zeros((n_edges, 1), dtype=np.float32)
    dynamic_edges = q1[:, :, None].astype(np.float32)

    # Reproduce mSWEGNNBoundaryCondition._init() without requiring its on-disk
    # simulation-path setup. This is the exact state consumed by the inherited
    # create/remove/apply methods in data.boundary_condition.BoundaryCondition.
    bc = BoundaryCondition.__new__(BoundaryCondition)
    bc.init_inflow_boundary_nodes = np.array([inflow_ghost], dtype=np.int64)
    bc.init_outflow_boundary_nodes = np.array([], dtype=np.int64)
    bc.ghost_nodes = ghost_nodes
    bc.boundary_nodes_mapping = {int(inflow_ghost): int(n_faces)}
    bc.new_inflow_boundary_nodes = np.array([n_faces], dtype=np.int64)
    bc.new_outflow_boundary_nodes = np.array([], dtype=np.int64)
    bc._is_called = {"create": False, "remove": False, "apply": False}
    bc._boundary_edge_index = None
    bc._boundary_dynamic_edges = None

    initial_edge_count = edge_index.shape[1]
    assert initial_edge_count == n_edges
    assert dynamic_edges.shape[1] == initial_edge_count

    bc.create(edge_index, dynamic_edges)
    (
        static_nodes,
        dynamic_nodes,
        static_edges,
        dynamic_edges,
        edge_index,
    ) = bc.remove(
        static_nodes,
        dynamic_nodes,
        static_edges,
        dynamic_edges,
        edge_index,
    )

    expected_internal = int(np.sum(np.all(edge_faces >= 0, axis=1)))
    assert edge_index.shape[1] == expected_internal, (
        f"After removal expected {expected_internal} internal edges, "
        f"got {edge_index.shape[1]}"
    )
    assert dynamic_edges.shape[1] == edge_index.shape[1]
    assert static_nodes.shape[0] == n_faces
    assert dynamic_nodes.shape[1] == n_faces

    (
        static_nodes,
        dynamic_nodes,
        static_edges,
        dynamic_edges,
        edge_index,
    ) = bc.apply(
        static_nodes,
        dynamic_nodes,
        static_edges,
        dynamic_edges,
        edge_index,
    )

    assert static_nodes.shape[0] == n_faces + 1
    assert dynamic_nodes.shape[1] == n_faces + 1
    assert edge_index.shape[1] == expected_internal + 1
    assert dynamic_edges.shape[1] == edge_index.shape[1]
    assert edge_index.max() < static_nodes.shape[0]
    assert int(bc.boundary_edges_mask.sum()) == 1
    assert int(bc.inflow_edges_mask.sum()) == 1

    print(f"PASS: {map_nc}")
    print(f"  NetCDF edges: {n_edges}")
    print(f"  Type-2 inflow edges: {len(type2_edges)}")
    print(f"  Type-3 wall edges: {len(type3_edges)}")
    print(f"  Final nodes: {static_nodes.shape[0]}")
    print(f"  Final edges: {edge_index.shape[1]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-nc", type=Path, required=True)
    args = parser.parse_args()
    run_test(args.map_nc)


if __name__ == "__main__":
    main()
