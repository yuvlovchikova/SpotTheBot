#!/bin/bash
#SBATCH -c 10
#SBATCH --time=720:00:00
#SBATCH --output=/home/yuvlovchikova/Yulia/output/TurkmenTF-IDF.log

source /home/yuvlovchikova/Yulia/my_env/bin/activate
python TurkmenTF-IDF.py
