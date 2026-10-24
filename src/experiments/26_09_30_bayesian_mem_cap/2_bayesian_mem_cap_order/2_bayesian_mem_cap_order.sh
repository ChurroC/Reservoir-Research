#!/bin/bash

#SBATCH --partition=mie_seara
#SBATCH --job-name=bayesian_job
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --time=07:00:00
#SBATCH --output=/home/charanc2/mie_seara_link/charanc2/Reservoir-Research/src/26_09_30_bayesian_mem_cap/2_bayesian_mem_cap_order/output/%j/output.log
#SBATCH --mail-user=charanc2@uic.edu

CODE_DIR="/home/charanc2/projects/Reservoir-Research/src/26_09_30_bayesian_mem_cap/2_bayesian_mem_cap_order"
OUTPUT_DIR="$CODE_DIR/output"
cd "$OUTPUT_DIR"

echo "Job started at: $(date +%Y/%m/%d_%H-%M-%S)"

uv run python -u "$CODE_DIR/2_bayesian_mem_cap_order.py"

echo "Job finished at: $(date +%Y/%m/%d_%H-%M-%S)"

