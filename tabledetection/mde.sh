#!/bin/bash
#SBATCH --mail-user=arnab.ghosh.chowdhury@uos.de
#SBATCH --mail-type=end,fail
#SBATCH --output=out-TL-TableDetection
#SBATCH --error=err-TL-TableDetection
#SBATCH --job-name="Matrix TL Table detection"


conda init bash
source ~/.bashrc
conda activate env_mde

echo "Script started !"
python src/train.py
echo "Script ended !"
