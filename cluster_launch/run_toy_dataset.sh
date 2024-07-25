#!/bin/bash
#SBATCH -J mlmi-sca
#SBATCH -A MLMI-ar2217-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mail-type=NONE
#SBATCH --array=1-4
#SBATCH --output=logs_toy_datasets/toy_datasets_%A_%a.out
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


config=/rds/user/ar2217/hpc-work/SCA/SCA_project/cluster_launch/toy_datasets_params.txt

traj=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
kernel=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
linear_run=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)
sigma_low_rank=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $config)
l2_=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $6}' $config)
iterations=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $7}' $config)

python /rds/user/ar2217/hpc-work/SCA/SCA_project/run_toy_datasets.py --traj $traj --kernel $kernel --linear_run $linear_run --sigma_low_rank $sigma_low_rank --l2_ $l2_ --iterations $iterations

