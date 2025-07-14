#!/bin/bash
#SBATCH --job-name=unet_train
#SBATCH --output=outputs/logs/unet_train_%j.log
#SBATCH --error=outputs/logs/unet_train_%j.err
#SBATCH --ntasks=1
#SBATCH --mem=32G              
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00        
#SBATCH --partition=gpu2080

# === Load basic modules ===
module load palma/2022b
module load GCCcore/12.2.0
module load Python/3.10.8

# === Clean PYTHONPATH to avoid ~/.local conflicts ===
unset PYTHONPATH
unset PYTHONUSERBASE

# === Set up temporary virtual environment ===
TMP_DIR=$(mktemp -d)
echo "📦 Erstelle temporäre venv unter: $TMP_DIR/torch_env"
python -m venv $TMP_DIR/torch_env
source $TMP_DIR/torch_env/bin/activate

## === Install dependencies ===
echo "⬇️ Installiere Abhängigkeiten"
pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir tifffile numpy matplotlib scikit-image tqdm scikit-learn \
    albumentations opencv-python-headless kornia 

# === Torch & CUDA Check ===
echo "✅ PyTorch & CUDA Check"
python -c "import torch; print('Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# === Start Training ===
echo "🚀 Starte Training"
srun --cpu-bind=none python final_model_train.py

# === Clean up temporary directory ===
echo "🧹 Lösche temporäre venv $TMP_DIR"
rm -rf $TMP_DIR

echo "✅ Job abgeschlossen"
