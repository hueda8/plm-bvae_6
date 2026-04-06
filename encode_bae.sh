#!/bin/bash
#PBS -q HGpuB      
#PBS -l select=1:ncpus=2:ngpus=2
#PBS -l walltime=12:00:00
#PBS -N opf_bae_HGpuB
#PBS -j oe

cd $PBS_O_WORKDIR

source /home/hueda8/projects/00_NEDO_VHH/scripts/setenv_wsMOQA.sh

# encode test set to binary vectors
python encode_file.py experiment_configs/binary.yml
