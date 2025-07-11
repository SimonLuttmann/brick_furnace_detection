#!/bin/bash
#SBATCH --job-name=u-net_kiln_arne
#SBATCH --partition=gpu2080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/t/tstraus2/brick_furnace_detection/arne/logs/kiln-%j.log
#SBATCH --error=/home/t/tstraus2/brick_furnace_detection/arne/logs/kiln-%j.err



module load palma/2023a GCC/12.3.0 OpenMPI/4.1.5 Python/3.11.3 
module load SciPy-bundle/2023.07 PyTorch/2.1.2-CUDA-12.1.1



cd /home/t/tstraus2/brick_furnace_detection
#cp /scratch/tmp/sluttman/Brick_Data_Train.zip .

#export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
#unzip -o -qq Brick_Data_Train.zip

echo "Starting python script..."

python3 -m arne.launch_script_adamw_adjusted_termination