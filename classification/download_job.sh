#!/bin/bash
#SBATCH --job-name=tcga_download
#SBATCH --partition=compute
#SBATCH --output=/scratch/spanda/tcga_brca_data_classification/download_%j.log
#SBATCH --error=/scratch/spanda/tcga_brca_data_classification/download_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00

# Load anaconda
module load anaconda3-2021.05-gcc-8.5.0-tzdgetu
source /apps/spack/opt/spack/linux-rocky8-zen/gcc-8.5.0/anaconda3-2021.05-tzdgetua6vg6dstl26ufvg42fgcrjv2u/etc/profile.d/conda.sh
conda activate tcga_env

# Set Python path
export PYTHONPATH=/scratch/spanda/py_packages_classification:$PYTHONPATH

# Go to scratch and run
cd $SCRATCH/tcga_brca_data_classification
python ~/classification/tcga_brca_downloader.py
