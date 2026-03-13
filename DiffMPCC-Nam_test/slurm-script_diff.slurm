#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:01:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module purge
module load anaconda/anaconda-2023.09

# REQUIRED for batch jobs
source "$CONDA_PREFIX/etc/profile.d/conda.sh"

# Activate YOUR env (by path)
conda activate /lustre/fs1/home/na202123/JAX

# Debug (keep this!)
which python
python --version

python main_casadi_sensitivity_HPC.py