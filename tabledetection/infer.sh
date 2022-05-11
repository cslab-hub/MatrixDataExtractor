#!/bin/bash
conda init bash
source ~/.bashrc
conda activate env_mde

echo "Script started !"
python src/infer.py
echo "Script ended !"
