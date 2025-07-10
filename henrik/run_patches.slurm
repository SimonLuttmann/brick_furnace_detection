#!/bin/bash
#SBATCH --job-name=patch_gen
#SBATCH --output=logs/patch_gen_%j.log
#SBATCH --error=logs/patch_gen_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00          # Maximal 1,5 Stunden
#SBATCH --partition=express      # Nutze die Express-Partition

# === Load modules ===
module load palma/2022b
module load GCCcore/12.2.0
module load Python/3.10.8

# === Clean PYTHONPATH to avoid ~/.local conflicts ===
unset PYTHONPATH
unset PYTHONUSERBASE

# === Set up temporary virtual environment ===
TMP_DIR=$(mktemp -d)
echo "📦 Erstelle temporäre venv unter: $TMP_DIR/patch_env"
python -m venv $TMP_DIR/patch_env
source $TMP_DIR/patch_env/bin/activate

# === Install dependencies ===
echo "⬇️ Installiere Python-Abhängigkeiten"
pip install --no-cache-dir tifffile numpy tqdm

# === Start Patch Generation ===
echo "🚀 Generiere Patches"
#srun --cpu-bind=none python generate_patches.py --input_dir /scratch/tmp/sluttman/Brick_Data_Train \
#                                                --output_dir /scratch/tmp/sluttman/Brick_Patches \
#                                               --patch_size 64 --stride 32 --max_patches_per_image 500

srun --cpu-bind=none python generate_patches.py \
    --input_dir /scratch/tmp/sluttman/Brick_Data_Train \
    --output_dir /scratch/tmp/sluttman/Brick_Patches_Filtered \
    --patch_size 64 --stride 32 --max_patches_per_image 500 \
    --filter_empty True --use_offset True

# === Clean up temporary venv ===
echo "🧹 Lösche temporäre venv $TMP_DIR"
rm -rf $TMP_DIR

echo "✅ Patch-Generierung abgeschlossen"
