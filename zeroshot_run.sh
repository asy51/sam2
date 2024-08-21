#!/bin/bash
#SBATCH --account=vxc204_aisc
#SBATCH --partition=aisc
#SBATCH --job-name=SAM2
#SBATCH --time=1-00:00:00
#SBATCH --array=0-31
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1

MODELS=(t s l b)
PROMPTS=(point pointneg mask box)
DIMS=("3" "2")

MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / 6))
PROMPT_INDEX=$(((SLURM_ARRAY_TASK_ID / 2) % 3))
DIM_INDEX=$((SLURM_ARRAY_TASK_ID % 2))

MODEL=${MODELS[$MODEL_INDEX]}
PROMPT=${PROMPTS[$PROMPT_INDEX]}
DIM=${DIMS[$DIM_INDEX]}

echo "Running configuration: MODEL=$MODEL, PROMPT=$PROMPT, DIMS=$DIM"
conda activate sam2
cd /home/asy51/repos/segment-anything-2/
srun python -u zeroshot.py --model $MODEL --prompt $PROMPT --dims $DIM
