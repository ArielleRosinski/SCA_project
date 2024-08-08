#!/bin/bash
#SBATCH -J mlmi-sca
#SBATCH -A MLMI-ar2217-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mail-type=NONE
#SBATCH --array=1-23
#SBATCH --output=logs_sca/sca_%A_%a.out
#SBATCH -p ampere

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment

# Commands for running JAX
module avail nvidia-cuda-toolkit
export LD_LIBRARY_PATH=~/.conda/envs/sca_env/lib:$LD_LIBRARY_PATH
export PATH=~/.conda/envs/sca_env/bin:$PATH


source /home/${USER}/.bashrc
conda activate sca_env
export OMP_NUM_THREADS=1


config=/rds/user/ar2217/hpc-work/SCA/SCA_project/cluster_launch/sca_params_train_test.txt

d=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
dataset_path=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
dataset_test_path=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)
save_path=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $config)
learning_rate=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $6}' $config)
iterations=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $7}' $config)

python /rds/user/ar2217/hpc-work/SCA/SCA_project/MC_Maze_linear_SCA_train_test.py --d $d --dataset_path $dataset_path --dataset_test_path $dataset_test_path --save_path $save_path --learning_rate $learning_rate --iterations $iterations

