#!/bin/bash
#SBATCH --cpus-per-task=24
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$your email here
#SBATCH --job-name=$your job name here
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=12G
#SBATCH --time=$ day-hr:min:sec
#SBATCH --output=$your path here

module load PYTHON/3.8.5

python '$python_file.py'