#!/bin/bash
#SBATCH -p gpu_h100_4
#SBATCH --qos=qos_gpu_h100
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem 64G
#SBATCH -t 0-12:00
#SBATCH --job-name=tcga_wsi
#SBATCH -o /scratch/spanda/tcga_brca_data_classification/wsi_%j.log
#SBATCH -e /scratch/spanda/tcga_brca_data_classification/wsi_%j.err
#SBATCH --gres=gpu:1

module load anaconda3-2021.05-gcc-8.5.0-tzdgetu
source /apps/spack/opt/spack/linux-rocky8-zen/gcc-8.5.0/anaconda3-2021.05-tzdgetua6vg6dstl26ufvg42fgcrjv2u/etc/profile.d/conda.sh
conda activate tcga_env

export PYTHONPATH=/scratch/spanda/py_packages_classification:$PYTHONPATH

cd /scratch/spanda/tcga_brca_data_classification
python ~/classification/preprocess_wsi.py
