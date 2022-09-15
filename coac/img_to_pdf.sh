#!/bin/bash
conda init bash
source ~/.bashrc
conda activate env_mde

echo "Script started !"
python src/img_to_pdf.py
echo "Script ended !"
