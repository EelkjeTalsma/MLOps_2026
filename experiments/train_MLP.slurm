#!/bin/bash
#SBATCH --job-name=pcam_mlp
#SBATCH --partition=gpu_course
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/home3/scur2380/MLOps_2026/experiments/logs/train_%j.out
#SBATCH --error=/gpfs/home3/scur2380/MLOps_2026/experiments/logs/train_%j.err

module load 2025
module load Python /3.13.1 −GCCcore−14.2.0
module load matplotlib /3.10.3 − gfbf −2025a

python -m venv /gpfs/home3/scur2380/MLOps_2026/venv
source /gpfs/home3/scur2380/MLOps_2026/venv/bin/activate
pip install -r /gpfs/home3/scur2380/MLOps_2026/requirements.txt
pip install -e .
cd ~/MLOps_2026/experiments

python train.py --config configs/train_config.yaml
