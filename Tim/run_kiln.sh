#!/bin/bash
#SBATCH --job-name=kiln-train
#SBATCH --partition=gpu2080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
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

# Setup pip cache in home directory (fallback if SLURM_TMPDIR not available)
export PIP_CACHE_DIR="${SLURM_TMPDIR:-$HOME/tmp}/.pip_cache"
mkdir -p "$PIP_CACHE_DIR"
echo "Using pip cache directory: $PIP_CACHE_DIR"

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

# Zusätzliche Pakete für data_loader.py und performance_analysis.py
echo "=== Installing additional dependencies ==="
python -m pip install --user tifffile==2023.7.10
python -m pip install --user scikit-learn==1.3.0
python -m pip install --user matplotlib==3.7.2 seaborn==0.12.2

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

python -c "
try:
    import tifffile
    print(f'✓ tifffile {tifffile.__version__}')
except Exception as e:
    print(f'✗ tifffile: {e}')
"

python -c "
try:
    import sklearn
    print(f'✓ scikit-learn {sklearn.__version__}')
except Exception as e:
    print(f'✗ scikit-learn: {e}')
"

python -c "
try:
    import matplotlib
    import seaborn
    print(f'✓ matplotlib {matplotlib.__version__}')
    print(f'✓ seaborn {seaborn.__version__}')
except Exception as e:
    print(f'✗ matplotlib/seaborn: {e}')
"

# ─── 6) Scratch & Training ─────────────────────────────────────────────────────
TMP=${SLURM_TMPDIR:-$TMPDIR}; mkdir -p "$TMP"; cd "$TMP"

# Kopiere alle notwendigen Dateien
echo "=== Copying project files ==="
cp ~/Brick_Data_Train.zip .
cp ~/DetectionV3_2.py .
# REMOVED: cp ~/data_loader.py .
# REMOVED: cp ~/utils.py .
# REMOVED: cp ~/performance_analysis.py .

export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
unzip -q Brick_Data_Train.zip

echo "------ Starte Training ------"
python DetectionV3_2.py
