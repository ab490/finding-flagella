#!/bin/bash
#SBATCH --job-name=cv                     # Job name
#SBATCH --output=../logs/cv_train_%j.out  # Path for the standard output file
#SBATCH --error=../logs/cv_train_%j.err   # Path for the error file

#SBATCH --mail-type=ALL                   # Email notification for all states
#SBATCH --mail-user=anobajaj@iu.edu       # Email address for notifications
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -A c01560
#SBATCH --mem=40G
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4


# Load required modules
# module load python/gpu/3.11.5

# activate venv
# source /N/scratch/anobajaj/v_envs/cv_env/bin/activate


# Run the Python script
# Pass the argument for model_name
MODEL_NAME=("yolov8n" "yolov8s" "yolov8m" "yolov8l" "yolov8x" "yolov9t" "yolov9s" "yolov9m" "yolov9c" "yolov9e")

for model_name in "${MODEL_NAME[@]}"; do
    echo "Running for model_name=$model_name" 
    python ../py_files/train_model.py "$model_name"
done

echo "All models trained!"
