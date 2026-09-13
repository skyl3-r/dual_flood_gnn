#!/bin/bash
#SBATCH --job-name=download_data
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G

set -e

DATASET_ROOT="$(cd "$(dirname "$0")/datasets" && pwd)"

# remove any unfinished zip downloads from the dataset directory
rm -f "$DATASET_ROOT"/raw_datasets_mesh.zip "$DATASET_ROOT"/raw_datasets_dk15.zip

curl -L --retry 5 --retry-delay 2 --retry-all-errors \
  -o "$DATASET_ROOT"/raw_datasets_mesh.zip \
  https://zenodo.org/api/records/13326595/files/raw_datasets_mesh.zip/content

curl -L --retry 5 --retry-delay 2 --retry-all-errors \
  -o "$DATASET_ROOT"/raw_datasets_dk15.zip \
  https://zenodo.org/api/records/13326595/files/raw_datasets_dk15.zip/content
