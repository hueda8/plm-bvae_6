#!/bin/bash
#PBS -q workq
#PBS -l select=1:ncpus=2:ngpus=2:mem=128gb:host=gpua01
#PBS -l walltime=200:00:00
#PBS -N opf_bae_ws01
#PBS -j oe

cd $PBS_O_WORKDIR

source /home/hueda8/projects/00_NEDO_VHH/scripts/setenv_plm_bvae_py312.sh

python train.py experiment_configs/binary.yml
