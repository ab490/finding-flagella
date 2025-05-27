#!/bin/bash
#SBATCH --job-name=cv     # Job name
#SBATCH --output=../logs/cv_predict_%j.out  # Path for the standard output file
#SBATCH --error=../logs/cv_predict_%j.err   # Path for the error file

#SBATCH --mail-type=ALL                 # Email notification for all states
#SBATCH --mail-user=anobajaj@iu.edu     # Email address for notifications
#SBATCH -p gpu-debug
#SBATCH --gpus-per-node=1
#SBATCH -A c01560
#SBATCH --mem=40G
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4


# Load required modules
# module load python/gpu/3.11.5

# activate venv
# source /N/scratch/anobajaj/v_envs/cv_env/bin/activate


# Run the Python script
# pass the argument for model_name
MODEL_NAME=("yolov9e")

for model_name in "${MODEL_NAME[@]}"; do
    echo "Running for model_name=$model_name" 
    python ../py_files/predict_model.py "$model_name"
done

echo "Done!"
