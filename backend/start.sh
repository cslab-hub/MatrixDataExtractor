#!/bin/bash
echo "Start MongoDB and ElasticSearch !"
systemctl start mongod
service elasticsearch start

echo "Start Conda Env !"
conda init bash
source ~/.bashrc
conda activate env_mde

echo "Start Django Server !"
python manage.py runserver
echo "Script ended !"
