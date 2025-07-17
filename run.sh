#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --account=puchalla
#SBATCH --time=3000:00
#SBATCH --mail-type=end
#SBATCH --mail-user=go9487@princeton.edu

cd BASDL
python run_sim.py
