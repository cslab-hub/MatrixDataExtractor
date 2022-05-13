#!/bin/bash
conda init bash
source ~/.bashrc
conda activate env_mde

echo "Script started !"
python src/pre_process.py
echo "Script ended !"
