#!/bin/bash
#PBS -q HGpuB
#PBS -l select=1:ncpus=2:ngpus=1:mem=64gb
#PBS -l walltime=10:00:00
#PBS -N opf_esm_B
#PBS -j oe

cd $PBS_O_WORKDIR

source /home/hueda8/projects/00_NEDO_VHH/scripts/setenv_plm_bvae_py312.sh

python embed_esm2.py --mode train \
        --hparams_path ./experiment_configs/binary.yml \
        --amp_dtype fp32

python embed_esm2.py --mode dev \
        --hparams_path ./experiment_configs/binary.yml \
        --amp_dtype fp32

python embed_esm2.py --mode test \
        --hparams_path ./experiment_configs/binary.yml \
        --amp_dtype fp32

