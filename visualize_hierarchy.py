"""Export every cached hierarchy level for inspection in QGIS."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString


def _hierarchy_levels(values) -> int:
    if "hierarchy_num_levels" in values:
        return int(values["hierarchy_num_levels"])
    return 1 if "cluster" in values else 0


def _cache_name(ratios: Optional[Sequence[float]]) -> Optional[str]:
    if ratios is None:
        return None
    encoded = "-".join(str(float(r)).replace(".", "p") for r in ratios)
    return f"hierarchy_v2_L{len(ratios)}_r{encoded}"


def _value(values, level: int, name: str):
    new_key = f"hierarchy_{level}_{name}"
    if new_key in values:
        return values[new_key]
    old_names = {
        "cluster": "cluster",
        "edge_index": "coarse_edge_index",
        "num_nodes": "num_supernodes",
    }
    return values[old_names[name]]


def _write_level(out_path, level, child_regions, cluster, edge_index, crs):
    child_regions = child_regions.copy()
    child_regions["parent_id"] = cluster
    # Keep only the grouping key and geometry. Otherwise fields such as the
    # child-level node_id survive dissolve and collide with the new parent ID.
    regions = child_regions[["parent_id", "geometry"]].dissolve(
        by="parent_id", as_index=False
    )
    regions = regions.rename(columns={"parent_id": "node_id"})
    regions["num_child_nodes"] = np.bincount(
        cluster, minlength=int(cluster.max()) + 1 if len(cluster) else 0
    )
    regions.to_file(out_path, layer=f"level{level}_regions", driver="GPKG")

    centroids = np.column_stack([
        regions.geometry.centroid.x.to_numpy(),
        regions.geometry.centroid.y.to_numpy(),
    ])
    nodes = gpd.GeoDataFrame(
        {"node_id": np.arange(len(centroids), dtype=np.int64)},
        geometry=gpd.points_from_xy(centroids[:, 0], centroids[:, 1]),
        crs=crs,
    )
    nodes.to_file(out_path, layer=f"level{level}_nodes", driver="GPKG")

    lines = [
        LineString([centroids[source], centroids[target]])
        for source, target in edge_index.T
    ]
    edges = gpd.GeoDataFrame(
        {"source": edge_index[0], "target": edge_index[1]},
        geometry=lines,
        crs=crs,
    )
    edges.to_file(out_path, layer=f"level{level}_edges", driver="GPKG")
    return nodes, regions


def export_event(event: int, out_path: Path, root: Path,
                 hierarchy_ratios: Optional[Sequence[float]] = None) -> None:
    processed = root / "processed"
    cache_name = _cache_name(hierarchy_ratios)
    if cache_name is not None:
        processed = processed / cache_name
    raw = root / "raw"
    static_path = processed / f"static_values_event_{event}.npz"
    if hierarchy_ratios is None and not static_path.exists():
        candidates = sorted(processed.glob(
            f"hierarchy_v2_*/static_values_event_{event}.npz"
        ))
        if len(candidates) == 1:
            static_path = candidates[0]
        elif len(candidates) > 1:
            raise FileNotFoundError(
                f"Multiple hierarchy caches contain event {event}; pass "
                "--hierarchy-ratios to select one."
            )
    if not static_path.exists():
        raise FileNotFoundError(f"Cached hierarchy not found: {static_path}")

    values = np.load(static_path)
    num_levels = _hierarchy_levels(values)
    if num_levels == 0:
        raise ValueError(f"No hierarchy found in {static_path}")
    edge_index = values["edge_index"].astype(np.int64)

    nodes_path = raw / "Geometry" / "Nodes" / f"nodes_{event}.shp"
    cells_path = raw / "Geometry" / "Cells" / f"cells_{event}.shp"
    nodes = gpd.read_file(nodes_path)
    cells = gpd.read_file(cells_path)

    retained = nodes[nodes["node_type"].isin([1, 2])].copy()
    retained = retained.reset_index(drop=False).rename(columns={"index": "raw_idx"})
    first_cluster = _value(values, 0, "cluster").astype(np.int64)
    if len(retained) != len(first_cluster):
        raise ValueError(
            f"Processed graph has {len(first_cluster)} nodes, but raw type 1/2 "
            f"nodes provide {len(retained)}."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    retained["node_id"] = np.arange(len(retained), dtype=np.int64)
    retained["is_boundary"] = retained["node_type"].eq(2)
    retained.to_file(out_path, layer="fine_nodes", driver="GPKG")

    retained_raw_indices = retained["raw_idx"].to_numpy()
    fine_cells = cells.iloc[retained_raw_indices].copy().reset_index(drop=True)
    fine_cells["node_id"] = np.arange(len(fine_cells), dtype=np.int64)
    fine_cells.to_file(out_path, layer="fine_cells", driver="GPKG")

    xy = np.column_stack([retained.geometry.x, retained.geometry.y])
    fine_lines = [LineString([xy[source], xy[target]]) for source, target in edge_index.T]
    fine_edges = gpd.GeoDataFrame(
        {"source": edge_index[0], "target": edge_index[1]},
        geometry=fine_lines,
        crs=nodes.crs,
    )
    fine_edges.to_file(out_path, layer="fine_edges", driver="GPKG")

    child_regions = fine_cells[["geometry"]].copy()
    child_centroids = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=nodes.crs
    )
    for level in range(num_levels):
        cluster = _value(values, level, "cluster").astype(np.int64)
        parent_edges = _value(values, level, "edge_index").astype(np.int64)
        parent_nodes, child_regions = _write_level(
            out_path, level + 1, child_regions, cluster, parent_edges, nodes.crs
        )

        # Cross-level lines are useful for validating assignments, but can be
        # hidden in QGIS when the graph is visually crowded.
        parent_xy = np.column_stack([
            parent_nodes.geometry.x.to_numpy(),
            parent_nodes.geometry.y.to_numpy(),
        ])
        cross_lines = [
            LineString([child_centroids.geometry.iloc[child], parent_xy[parent]])
            for child, parent in enumerate(cluster)
        ]
        cross_edges = gpd.GeoDataFrame(
            {"child": np.arange(len(cluster)), "parent": cluster},
            geometry=cross_lines,
            crs=nodes.crs,
        )
        cross_edges.to_file(
            out_path, layer=f"level{level}_to_level{level + 1}_edges", driver="GPKG"
        )
        child_centroids = parent_nodes

    print(f"Wrote {out_path}")
    print(f"  hierarchy levels: {num_levels}")
    print("  layers: fine_nodes, fine_cells, fine_edges, levelN_nodes, levelN_regions, levelN_edges")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=int, required=True, help="Event/run number, e.g. 1")
    parser.add_argument("--root", type=Path, default=Path("data_mswegnn/datasets"))
    parser.add_argument(
        "--hierarchy-ratios", type=float, nargs="+", default=None,
        help="Ratios identifying the processed hierarchy cache, e.g. 0.1 0.1",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output .gpkg path")
    args = parser.parse_args()
    export_event(args.event, args.out, args.root, args.hierarchy_ratios)


if __name__ == "__main__":
    main()
