#!/bin/bash
#SBATCH --job-name=tcga_download
#SBATCH --output=/home/spanda/siddhartha/logs/download_%j.out
#SBATCH --error=/home/spanda/siddhartha/logs/download_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute
#SBATCH --signal=B:TERM@300

set -o pipefail
PS1=""

module load anaconda3-2021.05-gcc-8.5.0-i5w2nbn
source /apps/spack-stable/opt/spack/linux-rocky8-zen/gcc-8.5.0/anaconda3-2021.05-i5w2nbnp5mvg3rvzxg4vzoylnbulv3fo/etc/profile.d/conda.sh
conda activate tcga_env

which python
python -c "import requests; print('requests OK')" || { echo "ENV BROKEN"; exit 1; }

export TCGA_DATA_ROOT=/home/spanda/siddhartha/data
export TCGA_STATE_ROOT=/home/spanda/siddhartha/state

mkdir -p /home/spanda/siddhartha/logs
cd /home/spanda/siddhartha/src

python -u download_tcga.py --shard ${SHARD:-0} --n-shards 6 --time-budget-hours 9.5
