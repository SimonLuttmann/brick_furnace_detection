#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=gpu2080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/home/s/sluttman/brick_furnace_detection/simon/logs/eval-%j.log
#SBATCH --error=/home/s/sluttman/brick_furnace_detection/simon/logs/eval-%j.err



module load palma/2023a GCC/12.3.0 OpenMPI/4.1.5 Python/3.11.3 
module load SciPy-bundle/2023.07 PyTorch/2.1.2-CUDA-12.1.1



cd /home/s/sluttman/brick_furnace_detection
#cp /scratch/tmp/sluttman/Brick_Data_Train.zip .

#export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
#unzip -o -qq Brick_Data_Train.zip

echo "Starting python script..."

python3 check_model_eval.py