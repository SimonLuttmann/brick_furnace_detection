#!/bin/bash
#SBATCH --job-name=kiln-train
#SBATCH --partition=gpu2080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=kiln-%j.log

# ─── 1) Module laden ─────────────────────────────────────────────────────────
module --force purge
module load palma/2023a
module load GCC/12.3.0 OpenMPI/4.1.5
module load Python/3.11.3                # Korrekte Python-Version für 2023a
module load SciPy-bundle/2023.07
module load PyTorch/2.1.2-CUDA-12.1.1

# ─── 2) Python Environment Setup ─────────────────────────────────────────────
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR=$SLURM_TMPDIR/.pip_cache
mkdir -p $PIP_CACHE_DIR

# ─── 3) Python Path für user-installierte Pakete ─────────────────────────────
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH

# ─── 4) Pakete installieren ─────────────────────────────────────────────────────
echo "=== Installing pip packages ==="
python -m pip install --user --upgrade pip

# Rasterio installieren
echo "=== Installing rasterio ==="
python -m pip install --user rasterio==1.3.9

echo "=== Installing pytorch-lightning ==="
python -m pip install --user pytorch-lightning==2.2.2 torchmetrics==1.4.0

# ─── 5) Test der Installation ─────────────────────────────────────────────────
echo "=== Testing module imports ==="
python -c "
try:
    import rasterio
    print(f'✓ rasterio {rasterio.__version__}')
except Exception as e:
    print(f'✗ rasterio: {e}')
"

python -c "
try:
    import torch
    print(f'✓ torch {torch.__version__}')
except Exception as e:
    print(f'✗ torch: {e}')
"

python -c "
try:
    import pytorch_lightning
    print(f'✓ pytorch_lightning {pytorch_lightning.__version__}')
except Exception as e:
    print(f'✗ pytorch_lightning: {e}')
"

# ─── 6) Scratch & Training ─────────────────────────────────────────────────────
TMP=${SLURM_TMPDIR:-$TMPDIR}; mkdir -p "$TMP"; cd "$TMP"

cp ~/Brick_Data_Train.zip ~/sentinel_kiln_detection_final.py .
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
unzip -q Brick_Data_Train.zip

echo "------ Starte Training ------"
python sentinel_kiln_detection_final.py
