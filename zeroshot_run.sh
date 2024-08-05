#!/bin/bash
#SBATCH --account=vxc204_aisc
#SBATCH --partition=aisc
#SBATCH --job-name=SAM2
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

VALUES=(t s l b)
VALUE=${VALUES[$SLURM_ARRAY_TASK_ID]}
NEGATIVE="-n"
echo $VALUE $NEGATIVE

# Activate your Python environment, if needed
conda activate sam2

# Run the Python script with the part number
cd /home/asy51/repos/segment-anything-2/
srun python -u zeroshot.py -m $VALUE $NEGATIVE