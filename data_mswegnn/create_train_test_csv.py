#!/usr/bin/env python3
"""Create mSWE-GNN train/test manifest CSVs from the raw Zenodo file layout.

Expected raw tree after unpacking the mesh or dk15 archive:
    data_mswegnn/datasets/raw/
        DEM/
        Geometry/
        Hydrograph/
        overview.csv
        Simulations/

This script writes repository-compatible manifest files directly under the
raw/ directory:
    data_mswegnn/datasets/raw/train.csv
    data_mswegnn/datasets/raw/test.csv

The output rows carry the six columns the repository loader asserts:
    Run_ID,
    Simulation_Filepath,
    Nodes_Shp_Filepath,
    Edges_Shp_Filepath,
    DEM_Filepath,
    Cells_Shp_Filepath,
    Hydrograph_Filepath

The split is hard-coded as requested:
    train rows: Run_ID 1..80
    test rows:  Run_ID 81..100
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd

DATASET_DIR = Path(__file__).resolve().parent / 'datasets'
RAW_DIR = DATASET_DIR / 'raw'
TRAIN_CSV = RAW_DIR / 'train.csv'
TEST_CSV = RAW_DIR / 'test.csv'

EVENT_FILE_KEYS = [
    'Simulation_Filepath',
    'Nodes_Shp_Filepath',
    'Edges_Shp_Filepath',
    'DEM_Filepath',
    'Cells_Shp_Filepath',
    'Hydrograph_Filepath',
]

GEOMETRY_SCHEMA = {
    'nodes': {'X', 'Y', 'Elevation1', 'node_type'},
    'edges': {'from_node', 'to_node', 'length', 'slope', 'edge_type', 'fc_length', 'nc_edge_id'},
    'cells': {'area_m2'},
}

def validate_geometry_schema(nodes_file: Path, edges_file: Path, cells_file: Path) -> None:
    """Fail during dataset preparation if geometry was made by an old converter."""
    files = {'nodes': nodes_file, 'edges': edges_file, 'cells': cells_file}
    missing = {}
    for kind, path in files.items():
        columns = set(gpd.read_file(path, rows=0).columns)
        absent = GEOMETRY_SCHEMA[kind] - columns
        if absent:
            missing[kind] = sorted(absent)
    if missing:
        details = ', '.join(f'{kind}: {", ".join(columns)}' for kind, columns in missing.items())
        raise ValueError(
            f'Geometry shapefiles are missing required columns ({details}). '
            'Regenerate Geometry with convert_mswegnn_mesh.py, then rerun this script.'
        )

def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f'Missing raw folder: {RAW_DIR}. Flatten the zip first so the raw tree is exactly data_mswegnn/datasets/raw/.')

    sim_dir = RAW_DIR / 'Simulations'
    geom_dir = RAW_DIR / 'Geometry'
    nodes_dir = geom_dir / 'Nodes'
    edges_dir = geom_dir / 'Edges'
    cells_dir = geom_dir / 'Cells'
    dem_dir = RAW_DIR / 'DEM'
    hydro_dir = RAW_DIR / 'Hydrograph'

    for required_dir in [sim_dir, nodes_dir, edges_dir, cells_dir, dem_dir, hydro_dir]:
        if not required_dir.exists():
            raise FileNotFoundError(f'Missing required directory: {required_dir}')

    rows = []
    for run_id in range(1, 101):
        sim_file = sim_dir / f'output_{run_id}_map.nc'
        nodes_file = nodes_dir / f'nodes_{run_id}.shp'
        edges_file = edges_dir / f'edges_{run_id}.shp'
        dem_file = dem_dir / f'DEM_{run_id}.xyz'
        cells_file = cells_dir / f'cells_{run_id}.shp'
        hydro_file = hydro_dir / f'Hydrograph_{run_id}.txt'

        for p in [sim_file, nodes_file, edges_file, dem_file, cells_file, hydro_file]:
            if not p.exists():
                raise FileNotFoundError(f'Missing expected file for run_id {run_id}: {p}')

        validate_geometry_schema(nodes_file, edges_file, cells_file)

        rows.append({
            'Run_ID': run_id,
            'Simulation_Filepath': str(sim_file.relative_to(RAW_DIR)),
            'Nodes_Shp_Filepath': str(nodes_file.relative_to(RAW_DIR)),
            'Edges_Shp_Filepath': str(edges_file.relative_to(RAW_DIR)),
            'DEM_Filepath': str(dem_file.relative_to(RAW_DIR)),
            'Cells_Shp_Filepath': str(cells_file.relative_to(RAW_DIR)),
            'Hydrograph_Filepath': str(hydro_file.relative_to(RAW_DIR)),
        })

    manifest = pd.DataFrame(rows)

    # Split as how MSWE-GNN paper did their train-test split
    # As seen in https://github.com/RBTV1/mSWE-GNN/blob/main/database/create_dataset.ipynb
    train_df = manifest[manifest['Run_ID'].between(1, 80)].copy()
    test_df = manifest[manifest['Run_ID'].between(81, 100)].copy()

    # Keep only the validated file-key order
    train_df = train_df[['Run_ID', *EVENT_FILE_KEYS]]
    test_df = test_df[['Run_ID', *EVENT_FILE_KEYS]]

    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print(f'Wrote {TRAIN_CSV} with {len(train_df)} events.')
    print(f'Wrote {TEST_CSV} with {len(test_df)} events.')

if __name__ == '__main__':
    main()
