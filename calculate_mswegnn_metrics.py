#!/usr/bin/env python3
"""Calculate additional mSWE-GNN metrics from saved test metric files.

Example:
    python calculate_mswegnn_metrics.py \
        --model-prefix HierarchicalDUALFloodGNN_2026-10-09_09-56-56

If --model-prefix is omitted, every matching test metric file is processed.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate additional mSWE-GNN metrics from saved test outputs."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory).",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=None,
        help="Directory containing *_test_metrics.npz files.",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=None,
        help="Test event summary CSV. Defaults to the available dataset test.csv.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: additional_mswegnn_metrics.csv).",
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        default=None,
        help=(
            "Model checkpoint prefix, for example "
            "HierarchicalDUALFloodGNN_2026-10-09_09-56-56. "
            "If omitted, process every matching metric file."
        ),
    )
    parser.add_argument(
        "--no-unit-discharge",
        action="store_true",
        help="Skip unit-discharge MAE calculation.",
    )
    return parser.parse_args()


def shp_values(filepath: Path, column: str) -> np.ndarray:
    try:
        import shapefile
    except ImportError as exc:
        raise ImportError(
            "Reading shapefiles requires pyshp. Install it with "
            "'python -m pip install pyshp'."
        ) from exc

    reader = shapefile.Reader(str(filepath))
    field_names = [
        field[0]
        for field in reader.fields
        if field[0] != "DeletionFlag"
    ]
    column_index = field_names.index(column)
    return np.asarray([
        record[column_index]
        for record in reader.iterRecords()
    ])


def csi(pred_depth: np.ndarray, target_depth: np.ndarray, threshold: float) -> float:
    pred_flooded = pred_depth > threshold
    target_flooded = target_depth > threshold
    true_positive = np.logical_and(pred_flooded, target_flooded).sum()
    false_positive = np.logical_and(pred_flooded, ~target_flooded).sum()
    false_negative = np.logical_and(~pred_flooded, target_flooded).sum()
    denominator = true_positive + false_positive + false_negative
    return float(true_positive / denominator) if denominator else np.nan


def resolve_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def resolve_test_csv(project_root: Path, requested: Path | None) -> Path:
    if requested is not None:
        return resolve_path(requested, project_root)

    candidates = [
        project_root / "data_mswegnn" / "datasets" / "raw" / "test.csv",
        project_root / "data_mswegnn" / "datasets" / "test.csv",
        project_root / "data_mswegnn" / "datasets" / "raw" / "archive" / "test.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find test.csv. Use --test-csv to specify its location."
    )


def relative_raw_path(value: object, raw_dir: Path) -> Path:
    # Event CSV paths may use Windows separators even when processed on Linux.
    value = str(value).replace("\\", "/")
    return raw_dir / Path(value)


def calculate_metrics(
    metrics_dir: Path,
    raw_dir: Path,
    test_csv: Path,
    output_csv: Path,
    model_prefix: str | None,
    unit_discharge: bool,
) -> pd.DataFrame:
    summary = pd.read_csv(test_csv)
    summary["Run_ID"] = summary["Run_ID"].astype(str)
    summary_by_run = summary.set_index("Run_ID")

    files = sorted(metrics_dir.glob("*_runid_*_test_metrics.npz"))
    if model_prefix is not None:
        files = [
            path for path in files
            if path.name.startswith(model_prefix + "_runid_")
        ]
    if not files:
        prefix_text = f" for model prefix {model_prefix!r}" if model_prefix else ""
        raise FileNotFoundError(
            f"No metric files found in {metrics_dir}{prefix_text}."
        )

    rows = []
    for path in files:
        match = re.search(r"_runid_(.+?)_test_metrics\.npz$", path.name)
        if match is None:
            continue
        run_id = match.group(1)
        if run_id not in summary_by_run.index:
            raise KeyError(
                f"Run ID {run_id!r} from {path.name} is not present in {test_csv}."
            )

        event = summary_by_run.loc[run_id]
        with np.load(path, allow_pickle=True) as data:
            # Saved node arrays already exclude boundary nodes, matching the tester.
            cell_area_all = shp_values(
                relative_raw_path(event["Cells_Shp_Filepath"], raw_dir),
                "area_m2",
            ).astype(float)
            node_types = shp_values(
                relative_raw_path(event["Nodes_Shp_Filepath"], raw_dir),
                "node_type",
            )
            cell_area = cell_area_all[node_types == 1]

            pred_volume = np.asarray(data["pred"], dtype=float).squeeze(-1)
            target_volume = np.asarray(data["target"], dtype=float).squeeze(-1)
            if pred_volume.shape[-1] != len(cell_area):
                raise ValueError(
                    f"{path.name}: {pred_volume.shape} does not match "
                    f"{len(cell_area)} non-boundary cell areas"
                )
            pred_depth = pred_volume / cell_area
            target_depth = target_volume / cell_area

            edge_pred = np.asarray(data["edge_pred"], dtype=float).squeeze(-1)
            edge_target = np.asarray(data["edge_target"], dtype=float).squeeze(-1)
            edge_mae = np.mean(np.abs(edge_pred - edge_target))

            edge_length_all = shp_values(
                relative_raw_path(event["Edges_Shp_Filepath"], raw_dir),
                "fc_length",
            ).astype(float)
            edge_types = shp_values(
                relative_raw_path(event["Edges_Shp_Filepath"], raw_dir),
                "edge_type",
            )
            edge_length = edge_length_all[edge_types != 3]
            if edge_pred.shape[-1] != len(edge_length):
                raise ValueError(
                    f"{path.name}: {edge_pred.shape} does not match "
                    f"{len(edge_length)} edge lengths"
                )
            unit_discharge_mae = np.mean(
                np.abs(edge_pred / edge_length - edge_target / edge_length)
            )

            rows.append({
                "run_id": run_id,
                "source_file": path.name,
                "depth_mae_m": np.mean(np.abs(pred_depth - target_depth)),
                "discharge_mae_m3_per_s": edge_mae,
                "unit_discharge_mae_m2_per_s": (
                    unit_discharge_mae if unit_discharge else np.nan
                ),
                "csi_005m": csi(pred_depth, target_depth, 0.05),
                "csi_030m": csi(pred_depth, target_depth, 0.30),
                "inference_time_s_per_timestep": float(data["inference_time"]),
            })

    if not rows:
        raise RuntimeError("No metric files matched the expected filename pattern.")

    results = pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)
    return results


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    metrics_dir = resolve_path(
        args.metrics_dir or Path("saved_metrics"), project_root
    )
    raw_dir = project_root / "data_mswegnn" / "datasets" / "raw"
    test_csv = resolve_test_csv(project_root, args.test_csv)
    output_csv = resolve_path(
        args.output_csv or Path("additional_mswegnn_metrics.csv"),
        project_root,
    )

    results = calculate_metrics(
        metrics_dir=metrics_dir,
        raw_dir=raw_dir,
        test_csv=test_csv,
        output_csv=output_csv,
        model_prefix=args.model_prefix,
        unit_discharge=not args.no_unit_discharge,
    )

    metric_columns = [
        "depth_mae_m",
        "discharge_mae_m3_per_s",
        "unit_discharge_mae_m2_per_s",
        "csi_005m",
        "csi_030m",
        "inference_time_s_per_timestep",
    ]
    summary_stats = pd.DataFrame({
        "mean": results[metric_columns].mean(),
        "sd": results[metric_columns].std(ddof=1).fillna(0.0),
    })

    print(f"Processed {len(results)} event files")
    print(f"Metrics written to: {output_csv}")
    print("\nPer-event metrics:")
    print(results.to_string(index=False))
    print("\nSummary statistics:")
    print(summary_stats.to_string())


if __name__ == "__main__":
    main()
