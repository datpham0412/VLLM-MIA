#!/bin/bash
#SBATCH --job-name=aug_dalle_mia
#SBATCH --output=aug_dalle_mia_%j.out
#SBATCH --error=aug_dalle_mia_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=milan-gpu

# Navigate to your directory
cd /fred/oz402/abir/VLLM-MIA/deepseek-vl-evaclip/DeepSeek-VL/

# Create output directory for augmented DALL-E results
mkdir -p /fred/oz402/abir/VLLM-MIA/deepseek-vl-evaclip/DeepSeek-VL/Results/augmented_dalle_evaluation

# Use the exact Python path with augmented DALL-E script
/home/mabir/.conda/envs/deepseek-evac/bin/python /fred/oz402/abir/VLLM-MIA/deepseek-vl-evaclip/DeepSeek-VL/run_augmented_dalle_mia.py

echo "Augmented DALL-E evaluation completed"
