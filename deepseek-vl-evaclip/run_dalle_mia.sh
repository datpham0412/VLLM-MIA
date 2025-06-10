#!/bin/bash
#SBATCH --job-name=dalle_mia
#SBATCH --output=dalle_mia_%j.out
#SBATCH --error=dalle_mia_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=milan-gpu

# Navigate to your directory - UPDATED PATH
cd /fred/oz402/abir/VLLM-MIA/deepseek-vl-evaclip/DeepSeek-VL/

# Use the exact Python path with correct script path
/home/mabir/.conda/envs/deepseek-evac/bin/python /fred/oz402/abir/VLLM-MIA/deepseek-vl-evaclip/DeepSeek-VL/run_dalle_mia_complete.py

echo "DALL-E evaluation completed"