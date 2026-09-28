#!/bin/bash
 
#SBATCH --partition=mie_seara
#SBATCH --job-name=basic_bayesian_test
#SBATCH --nodes=1
#SBATCH --tasks-per-node=64
##SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/charanc2/projects/Reservoir-Research/src/week6/1_meta/simple_bayesian_job/output/log_%j.log
#SBATCH --mail-user=charanc2@uic.edu

CODE_DIR="/home/charanc2/projects/Reservoir-Research/src/week6/1_meta/simple_bayesian_job"
OUTPUT_DIR="/home/charanc2/projects/Reservoir-Research/src/week6/1_meta/simple_bayesian_job/output"

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo $(date +%Y/%m/%d_%H-%M-%S)
uv run python -u "$CODE_DIR/simple_bayesian_script.py"