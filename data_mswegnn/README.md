# mSWE-GNN dataset setup

This folder contains the mSWE-GNN dataset download and conversion helpers used to prepare the mesh files needed by the DUALFloodGNN training pipeline.

The expected dataset tree is:

```text
data_mswegnn/
  datasets/
    raw/
      Simulations/
      Geometry/
      DEM/
      Hydrograph/
      overview.csv
      train.csv
      test.csv
```

## 1. Download the mesh archive

From the repository root, enter the mSWE-GNN data helper folder:

```bash
cd data_mswegnn
```

Run the download helper:

```bash
# On a Slurm cluster:
sbatch download_data.sh

# Or, if you are just running it locally:
bash download_data.sh
```

This downloads the two Zenodo archives into `data_mswegnn/datasets/`:

- `raw_datasets_mesh.zip`
- `raw_datasets_dk15.zip`

The `dk15` archive is not needed for the mesh conversion workflow here, so you can ignore it for the first pass.

## 2. Unzip the mesh archive into the raw directory

The repository expects the extracted dataset files to land under `data_mswegnn/datasets/raw/`.

Use:

```bash
mkdir -p datasets/raw
unzip -o datasets/raw_datasets_mesh.zip -d datasets/raw
```

If the archive contains an extra top-level folder, flatten it once so that the directories `Simulations/`, `Geometry/`, `DEM/`, and `Hydrograph/` appear directly under `datasets/raw/`:

```bash
cp -r datasets/raw/raw_datasets_mesh/* datasets/raw/
rm -rf datasets/raw/raw_datasets_mesh
```

## 3. Convert NetCDF meshes to shapefiles

The converter always uses the historical mSWE-GNN loader workflow. It reads
the standard `Simulations/`, `DEM/`, and `Hydrograph/` folders and writes to
`Geometry/` automatically.

Run the conversion commands below from this `data_mswegnn/` directory.

For one run, provide only its ID:

```bash
python convert_mswegnn_mesh.py --run-id 1
```

For each run, the NetCDF simulation, DEM, and hydrograph files are required:

```text
Simulations/output_<run_id>_map.nc
DEM/DEM_<run_id>.xyz
Hydrograph/Hydrograph_<run_id>.txt
```

The converter uses the NetCDF file for the mesh and edge connectivity, the DEM
for cell elevations and terrain features, and the hydrograph to identify and
verify the type-2 inflow boundary.

For the full mesh suite, use a run-ID range:

```bash
python convert_mswegnn_mesh.py \
  --start-run-id 1 \
  --end-run-id 100
```

Each run produces:

```text
Geometry/
  Nodes/nodes_<run_id>.shp
  Edges/edges_<run_id>.shp
  Cells/cells_<run_id>.shp
```

Type-2 edges are verified against the hydrograph and represented as inflow
ghost edges. Type-3 edges are represented as wall ghosts and removed by the
historical boundary-condition pipeline. All NetCDF edges are exported in
`mesh2d_q1` order so the original loader can consume them directly.

The converter also writes mass-balance and inflow-detection diagnostics in the
geometry output directory. The expected inflow-edge count can be changed with
`--expected-inflow-edges`; use `-1` to disable that dataset-specific check.

## 4. Generate train/test CSV manifests

Once the extracted raw files are present and the geometry conversion has been run, create the train/test manifest files:

```bash
python create_train_test_csv.py
```

This writes:

- `datasets/raw/train.csv`
- `datasets/raw/test.csv`

Those files are the manifest files the model dataset readers expect.

## 5. Verify and train

Run the boundary-pipeline regression test on representative runs:

```bash
python test_original_boundary_pipeline.py \
  --map-nc datasets/raw/Simulations/output_1_map.nc
```

Return to the repository root before training:

```bash
cd ..
python train.py --config configs/mswegnn_config.yaml --model DUALFloodGNN
```

### Optional DEM coverage check

Before training, verify that the generated aspect rasters cover the real mesh
face centres. Ghost nodes are excluded from this check:

```bash
python data_mswegnn/check_dem_coverage.py \
  --start-run-id 1 \
  --end-run-id 100 \
  --require-all
```

The command reports the count and percentage of out-of-bounds face centres per
run and overall. To enforce a threshold, add for example:

```bash
python data_mswegnn/check_dem_coverage.py \
  --start-run-id 1 \
  --end-run-id 100 \
  --require-all \
  --max-percent 1.0
```
