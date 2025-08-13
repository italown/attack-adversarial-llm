#!/bin/bash
#SBATCH -p long
#SBATCH --job-name=nano-gcg
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=multiple_gcg_logs/gcg_vicuna_long_%j.out
#SBATCH --error=multiple_gcg_logs/gcg_vicuna_long_%j.err

mkdir -p multiple_gcg_logs

nvidea-smi

# Ativa conda
source ~/deteccao/llm-attacks/detec/bin/activate

cd ~/deteccao/attack-adversarial-llm

python -u harmful_strings.py