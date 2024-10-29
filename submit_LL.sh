#!/bin/bash

#SBATCH --job-name=LebwohlLasherProject
#SBATCH --partition=teach_cpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=28
#SBATCH --cpus-per-task=1
#SBATCH --time=0:0:10
#SBATCH --mem-per-cpu=100M
#SBATCH --account=PHYS033185

# Direct output to the following files.
# (The %j is replaced by the job id.)
SBATCH -o LLMPI_out_%j.txt

# Just in case this is not loaded already...
#module load languages/miniforge3/2020-3.8.5
module  add languages/python/3.12.3

# Change to working directory, where the job was submitted from.
cd "${SLURM_SUBMIT_DIR}"

# Record some potentially useful details about the job: 
echo "Running on host $(hostname)"
echo "Started on $(date)"
echo "Directory is $(pwd)"
echo "Slurm job ID is ${SLURM_JOBID}"
echo "This jobs runs on the following machines:"
echo "${SLURM_JOB_NODELIST}" 
printf "\n\n"

# Submit
# python LebwohlLasher.py 50 50 0.5 0

mpiexec -n 16 python ./LebwohlLasherMPI_ver2.py 50 50 0.5 0

# Output the end time
printf "\n\n"
echo "Ended on: $(date)"
