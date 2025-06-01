#!/bin/bash
#SBATCH --job-name=aug_flickr_mia
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=aug_flickr_mia_output_%j.out
#SBATCH --error=aug_flickr_mia_error_%j.err

# Load required modules
module load mamba

# Activate your environment
mamba activate deepseek-evac

# Navigate to your working directory
cd /fred/oz402/abir/VLLM-MIA/deepseek-vl-evaclip/DeepSeek-VL

# Copy utility files from group's VLLM-MIA directory (if needed)
cp /fred/oz402/abir/VLLM-MIA/metric_util.py .
cp /fred/oz402/abir/VLLM-MIA/eval.py .

# Create output directories
mkdir -p ./Results/augmented_flickr_evaluation/gen_32_tokens

# Run the MIA attack on augmented Flickr dataset
echo "Starting DeepSeek-VL + EVA-CLIP MIA attack on augmented Flickr..."
python run_deepseek_mia_fixed.py \
    --gpu_id 0 \
    --num_gen_token 32 \
    --dataset /fred/oz402/aho/VLLM-MIA/Data/augmented_Flickr \
    --output_dir "./Results/augmented_flickr_evaluation" \
    --checkpoint_every 20

echo "Augmented Flickr evaluation completed!"
