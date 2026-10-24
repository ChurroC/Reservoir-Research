#!/bin/bash

#SBATCH --partition=mie_seara_gpu
#SBATCH --job-name=bayesian_job
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=00:10:00
#SBATCH --output=/home/charanc2/projects/Reservoir-Research/src/week7/1_test_if_working/bayesian_job/output/%j/output.log
#SBATCH --mail-user=charanc2@uic.edu

CODE_DIR="/home/charanc2/projects/Reservoir-Research/src/week7/1_test_if_working/bayesian_job"
OUTPUT_DIR="$CODE_DIR/output"
cd "$OUTPUT_DIR"

echo "Job started at: $(date +%Y/%m/%d_%H-%M-%S)"

uv run python -u "$CODE_DIR/mult_bayesian_job.py"

echo "Job finished at: $(date +%Y/%m/%d_%H-%M-%S)"