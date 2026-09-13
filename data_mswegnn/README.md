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

## 3. Convert NetCDF mesh files to shapefile geometry

The mesh converter in `convert_mswegnn_mesh.py` supports two workflows:

- a single-file call
- a batch conversion over a run-ID range

For the full mesh suite, prefer the batch form:

```bash
python convert_mswegnn_mesh.py \
  --input-dir datasets/raw/Simulations \
  --output datasets/raw/Geometry \
  --start-run-id 1 \
  --end-run-id 100
```

That converts each `output_{run_id}_map.nc` into the required `Nodes/`, `Edges/`, and `Cells/` shapefile folders under the chosen output root. The converter writes one output set per run ID.

You can also use the single-file invocation if you only need one simulation:

```bash
python convert_mswegnn_mesh.py \
  --input datasets/raw/Simulations/output_1_map.nc \
  --output datasets/raw/Geometry \
  --run-id 1
```

## 4. Generate train/test CSV manifests

Once the extracted raw files are present and the geometry conversion has been run, create the train/test manifest files:

```bash
python create_train_test_csv.py
```

This writes:

- `datasets/raw/train.csv`
- `datasets/raw/test.csv`

Those files are the manifest files the model dataset readers expect.

## 5. Train DUALFloodGNN

After the raw archive has been expanded into `datasets/raw/`, the geometry files have been produced, and the train/test CSVs have been created, you can proceed with the DUALFloodGNN training command from the repository root.
