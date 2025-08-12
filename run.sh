#!/bin/bash
#SBATCH -p long 
#SBATCH --job-name=nano-gcg
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=multiple_gcg_logs/gcg_vicuna_long_%j.out
#SBATCH --error=multiple_gcg_logs/gcg_vicuna_long_%j.err

# Ativa conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate treino

# Entra no diretório do projeto
cd ~/llm-attacks/experiments/launch_scripts

# (Opcional) Exporta PYTHONPATH explicitamente
export PYTHONPATH=$PYTHONPATH:/home/CIN/rrl3/llm-attacksS

python main.py