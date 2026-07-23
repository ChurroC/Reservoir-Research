#!/bin/bash

#SBATCH --partition=mie_seara
#SBATCH --job-name=bayesian_job
#SBATCH --nodes=1
#SBATCH --tasks-per-node=10
## SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=/home/charanc2/projects/Reservoir-Research/src/week7/1_test_if_working/bayesian_job/output/log_%j.log
#SBATCH --mail-user=charanc2@uic.edu

CODE_DIR="/home/charanc2/projects/Reservoir-Research/src/week7/1_test_if_working/bayesian_job"
OUTPUT_DIR="$CODE_DIR/output"

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "Job started at: $(date +%Y/%m/%d_%H-%M-%S)"

uv run python -u "$CODE_DIR/bayesian_job.py"

echo "Job finished at: $(date +%Y/%m/%d_%H-%M-%S)"