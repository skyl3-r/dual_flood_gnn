#!/usr/bin/env python3
"""Check that generated DEM rasters cover the real mesh-face centres."""

import argparse
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.transform import rowcol


def check_run(raw_dir: Path, run_id: int) -> tuple[int, int] | None:
    nodes_path = raw_dir / "Geometry" / "Nodes" / f"nodes_{run_id}.shp"
    dem_path = raw_dir / "DEM" / f"DEM_{run_id}_aspect.tif"

    if not nodes_path.exists() or not dem_path.exists():
        print(f"run {run_id}: skipped; missing nodes or DEM raster")
        return None

    nodes = gpd.read_file(nodes_path)
    if "node_type" in nodes.columns:
        nodes = nodes[nodes["node_type"] == 1]

    with rasterio.open(dem_path) as src:
        rows, cols = rowcol(
            src.transform,
            nodes["X"].to_numpy(),
            nodes["Y"].to_numpy(),
        )
        outside = (
            (rows < 0)
            | (rows >= src.height)
            | (cols < 0)
            | (cols >= src.width)
        )

    count = int(outside.sum())
    total = len(nodes)
    percentage = 100 * count / total if total else 0.0
    print(f"run {run_id}: {count}/{total} out of bounds ({percentage:.4f}%)")
    return count, total


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check generated mSWE-GNN DEM coverage against mesh faces."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data_mswegnn/datasets/raw"),
        help="mSWE-GNN raw data directory",
    )
    parser.add_argument("--start-run-id", type=int, default=1)
    parser.add_argument("--end-run-id", type=int, default=100)
    parser.add_argument(
        "--max-percent",
        type=float,
        default=None,
        help="Fail if any checked run exceeds this percentage",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any requested run is missing nodes or DEM raster",
    )
    args = parser.parse_args()

    if args.start_run_id > args.end_run_id:
        parser.error("--start-run-id must be <= --end-run-id")

    total_outside = 0
    total_nodes = 0
    missing = 0
    exceeded = []

    for run_id in range(args.start_run_id, args.end_run_id + 1):
        result = check_run(args.raw_dir, run_id)
        if result is None:
            missing += 1
            continue
        count, total = result
        total_outside += count
        total_nodes += total
        if args.max_percent is not None and total:
            if 100 * count / total > args.max_percent:
                exceeded.append(run_id)

    if total_nodes:
        print("\nOverall:")
        print(f"out of bounds: {total_outside}/{total_nodes}")
        print(f"percentage: {100 * total_outside / total_nodes:.4f}%")

    if args.require_all and missing:
        print(f"\nFailed: {missing} requested run(s) are incomplete.")
        return 1
    if exceeded:
        print(f"\nFailed: percentage threshold exceeded by run(s): {exceeded}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
