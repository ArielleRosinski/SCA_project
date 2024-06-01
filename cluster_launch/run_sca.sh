#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-2
#SBATCH --job-name=rnn_run
#SBATCH --output=logs_sca/sca_%A_%a.out


config=/rds/user/ar2217/hpc-work/SCA

d=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)

module load python
python /rds/user/ar2217/hpc-work/SCA/SCA_project/MC_Maze_linear_SCA.py $d 

