#!/bin/bash
#SBATCH -p v100
#SBATCH -q gpu
#SBATCH -c 4
#SBATCH --mem 140G
#SBATCH --gres=gpu:1
gpu_='0'         #gpu_='1,2,3'

root_path='/storage/wanyihanLab/sunrui03/20240120_H9_V6_Day3_DL/0120raw/'    #rootpath去掉最后一层

source /home/wanyihanLab/sunrui03/anaconda3/bin/activate cell-track
python Creatdatafloder.py $gpu_ $root_path    ## 获取当前目录

