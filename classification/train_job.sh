#!/bin/bash
#SBATCH --job-name=tcga_train
#SBATCH --partition=compute
#SBATCH --output=/scratch/spanda/tcga_brca_data_classification/train_%j.log
#SBATCH --error=/scratch/spanda/tcga_brca_data_classification/train_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

module load anaconda3-2021.05-gcc-8.5.0-tzdgetu
source /apps/spack/opt/spack/linux-rocky8-zen/gcc-8.5.0/anaconda3-2021.05-tzdgetua6vg6dstl26ufvg42fgcrjv2u/etc/profile.d/conda.sh
conda activate tcga_env
export PYTHONPATH=/scratch/spanda/py_packages_classification:$PYTHONPATH

cd /scratch/spanda/tcga_brca_data_classification
python ~/classification/train_classifier.py
