#!/bin/bash
#SBATCH --job-name=tcga_extract
#SBATCH --output=/home/spanda/siddhartha/logs/extract_%j.out
#SBATCH --error=/home/spanda/siddhartha/logs/extract_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=compute
#SBATCH --signal=B:TERM@300

set -o pipefail
PS1=""

module load anaconda3-2021.05-gcc-8.5.0-i5w2nbn
source /apps/spack-stable/opt/spack/linux-rocky8-zen/gcc-8.5.0/anaconda3-2021.05-i5w2nbnp5mvg3rvzxg4vzoylnbulv3fo/etc/profile.d/conda.sh
conda activate tcga_env

which python
python -c "import torch, openslide, timm, transformers; print('deps OK')" || { echo "ENV BROKEN"; exit 1; }

export TCGA_DATA_ROOT=/home/spanda/siddhartha/data
export TCGA_STATE_ROOT=/home/spanda/siddhartha/state
export HF_HOME=/home/spanda/.cache/huggingface

mkdir -p /home/spanda/siddhartha/logs
cd /home/spanda/siddhartha/src

python -u extract_all_backbones.py --shard ${SHARD:-0} --n-shards ${N_SHARDS:-1} --time-budget-hours 70
