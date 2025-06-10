#!/bin/bash
#SBATCH --job-name=deepseek-eva-mia
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=deepseek_mia_output_%j.out
#SBATCH --error=deepseek_mia_error_%j.err

# Load required modules
module load mamba

# Activate your environment
mamba activate deepseek-evac

# Navigate to your working directory
cd /fred/oz402/abir/VLLM-MIA/deepseek-vl-evaclip/DeepSeek-VL

# Copy utility files from group's VLLM-MIA directory
cp /fred/oz402/abir/VLLM-MIA/metric_util.py .
cp /fred/oz402/abir/VLLM-MIA/eval.py .

# Create output directories
mkdir -p ./Results/gen_32_tokens

# Run the MIA attack
echo "Starting DeepSeek-VL + EVA-CLIP MIA attack..."
python run_deepseek_mia_fixed.py \
    --gpu_id 0 \
    --num_gen_token 32 \
    --dataset /fred/oz402/abir/VLLM-MIA/Data/img_Flickr \
    --output_dir "./Results" \
    --checkpoint_every 20
echo "Job completed!"