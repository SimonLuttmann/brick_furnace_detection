#!/bin/bash
#SBATCH --job-name=kiln-v4-2
#SBATCH --output=/home/t/tstraus2/kiln-%j-v4-2.log
#SBATCH --error=/home/t/tstraus2/kiln-%j-v4-2.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

echo "🚀 ENHANCED UNET V4.2 - REVOLUTIONARY ARCHITECTURE TRAINING"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo "=============================================================="

# Enhanced Environment Setup
module load python/3.9
module load cuda/11.8
module load cudnn/8.9.2

# Activate virtual environment
source /home/t/tstraus2/venv/bin/activate

# Enhanced GPU Configuration
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
export CUDA_LAUNCH_BLOCKING=0  # Async for performance

# Memory Management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=8

# Enhanced Logging
echo "🔧 Environment Configuration:"
echo "   Python: $(python --version)"
echo "   PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "   CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "   GPU Count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "   GPU Name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"
echo "   Memory: $(free -h | grep Mem)"
echo "=============================================================="

# Change to project directory
cd /home/t/tstraus2/PycharmProjects/furnlDetection/brick_furnace_detection

echo "🎯 Starting Enhanced UNet V4.2 Training..."
echo "Expected Improvements (KILN CLASSES ONLY):"
echo "   • Macro F1 (Kiln): 0.77 → 0.85+ (+10%) - Background excluded"
echo "   • Rare classes IoU: 0.32-0.36 → 0.50+ (+40%)"
echo "   • Kiln-classes Precision: Significantly improved"
echo "   • Training stability: Significantly improved"
echo "   • Background (orig. 0) mapped to ignore_index, ALL 8 kiln classes in F1"
echo "=============================================================="

# Run the enhanced training script
python Tim/DetectionV4_2.py

echo "=============================================================="
echo "🎯 Enhanced UNet V4.2 Training Completed!"
echo "Time: $(date)"
echo "Job ID: $SLURM_JOB_ID completed"
echo "==============================================================" 