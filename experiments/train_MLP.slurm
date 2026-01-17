#!/bin/bash
#SBATCH --job-name=training1
#SBATCH --partition=gpu_course
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiments/logs/train_%j.out
#SBATCH --error=experiments/logs/train_%j.err

set -euo pipefail

mkdir -p experiments/logs

source ~/venv/bin/activate

echo "Node: $SLURMD_NODENAME"
nvidia-smi

python -u - <<EOF
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
print("Torch CUDA:", torch.version.cuda)
EOF


# module load 2025
# module load Python/3.13.1 −GCCcore−14.2.0
# module load matplotlib/3.10.3−gfbf−2025a

source ~/venv/bin/activate
# python -m venv /gpfs/home3/scur2380/MLOps_2026/venv
# source /gpfs/home3/scur2380/MLOps_2026/venv/bin/activate
# pip install -e .
# pip install -r /gpfs/home3/scur2380/MLOps_2026/requirements.txt
# cd ~/MLOps_2026/experiments
cd ~/MLOps_2026

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -u experiments/train.py --config experiments/configs/train_config.yaml
