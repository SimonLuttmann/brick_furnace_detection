#!/bin/bash
#SBATCH --job-name=kiln-v4-final
#SBATCH --output=/home/t/tstraus2/kiln-%j-v4-final.log
#SBATCH --error=/home/t/tstraus2/kiln-%j-v4-final.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

echo "🚀 ENHANCED UNET V4 FINAL - PRODUCTION READY TRAINING"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo "=============================================================="

# Environment Setup
module load python/3.9
module load cuda/11.8
module load cudnn/8.9.2

# Activate virtual environment
source /home/t/tstraus2/venv/bin/activate

# GPU Configuration
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
export CUDA_LAUNCH_BLOCKING=0

# Memory Management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=8

# Environment Info
echo "🔧 Environment Configuration:"
echo "   Python: $(python --version)"
echo "   PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "   CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "   GPU Count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "   GPU Name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"
echo "=============================================================="

# Change to project directory
cd /home/t/tstraus2/PycharmProjects/furnlDetection/brick_furnace_detection

echo "🎯 Starting Enhanced UNet V4 Final Training..."
echo "Architecture Features:"
echo "   • Attention Gates for rare class focus"
echo "   • Residual connections for gradient stability" 
echo "   • Combined Focal + Dice Loss (60/40)"
echo "   • AdamW optimizer with cosine annealing"
echo "   • Adaptive sampling for extreme class imbalance"
echo "   • 8 Kiln classes (background excluded from F1)"
echo "=============================================================="

# Run the final training script
python Tim/DetectionV4_final.py

echo "=============================================================="
echo "🎯 Enhanced UNet V4 Final Training Completed!"
echo "Time: $(date)"
echo "Job ID: $SLURM_JOB_ID completed"
echo "==============================================================" 