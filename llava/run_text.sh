#!/bin/bash

# Define script path
script="/fred/oz402/aho/VLLM-MIA/run_with_text.py"

# Extract base name without extension for job name
base_name=$(basename "$script" .py)

# Submit the job using a Slurm script
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=llava_${base_name}               # Job name
#SBATCH --output=${base_name}_job_output_%j.log     # Output log file (%j will be replaced with the job ID)
#SBATCH --error=${base_name}_job_error_%j.log       # Error log file (%j will be replaced with the job ID)
#SBATCH --ntasks=1                                  # Number of tasks (usually 1 for a single script)
#SBATCH --cpus-per-task=12                          # Number of CPU cores per task
#SBATCH --gres=gpu:1                                # Request 1 GPU
#SBATCH --mem=16G                                   # Memory per node
#SBATCH --time=7:00:00                              # Maximum execution time
#SBATCH --partition=gpu                             # Submit to GPU partition

# Load required modules
module load cuda/11.7.0
module -q load conda

# Activate your Conda environment using Mamba: run "conda env list" to check
mamba activate llava

# Navigate to the project directory
cd /fred/oz402/aho/VLLM-MIA

# Run the Python script
python run_with_text.py --gpu_id 0 --text_len 32 --dataset /fred/oz402/aho/VLLM-MIA/Data/llava_v15_gpt_text
EOT
