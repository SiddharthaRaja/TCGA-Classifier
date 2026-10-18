#!/bin/bash
#SBATCH --job-name=tcga_ext_big
#SBATCH --output=/home/spanda/siddhartha/logs/extract_big_%j.out
#SBATCH --error=/home/spanda/siddhartha/logs/extract_big_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --mem=240G
#SBATCH --cpus-per-task=128
#SBATCH --partition=big_compute
#SBATCH --signal=B:TERM@300

# big_compute enforces MinTRES cpu=128 per job, so one job runs 16 CPU
# extraction shards side by side. Per-patient locks stop them colliding.

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

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
for s in $(seq 0 15); do
  python -u extract_all_backbones.py --shard $s --n-shards 16 --time-budget-hours 46 \
    > /home/spanda/siddhartha/logs/ext_${SLURM_JOB_ID}_shard${s}.log 2>&1 &
done
wait
