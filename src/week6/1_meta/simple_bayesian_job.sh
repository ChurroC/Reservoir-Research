#!/bin/bash
 
#SBATCH --partition=mie_seara
#SBATCH --job-name=basic_bayesian
#SBATCH --nodes=1
#SBATCH --tasks-per-node=20
##SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=myjob_%j.log
#SBATCH --mail-user=charanc2@uic.edu

uv run python -u ./simple_bayesian_script.py