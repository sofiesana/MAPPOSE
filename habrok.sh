#!/bin/bash
#SBATCH --job-name=cheetah_search         # Job name
#SBATCH --output=cheetah-search-job-%j.log
#SBATCH --nodes=1                     # Number of nodes (use 1 node)
#SBATCH --ntasks=1                    # One task
#SBATCH --gpus-per-node=v100:1              
#SBATCH --mem=10GB                     # Total memory for the job (adjust based on need)
#SBATCH --time=30:00:00              # Time limit for the job (e.g., 2 hours)

# remove all previously loaded modules
module purge

# load python 3.8.16
module load Python/3.11.5-GCCcore-13.2.0  
 
# activate virtual environment
source $HOME/venvs/mappose/bin/activate

# TO DO
# mkdir -p /scratch/s4716671/MARL/search_results/pendulum

############ GETTING THE CODE
mkdir -p $TMPDIR

# copy code into TMPDIR
cp -r /scratch/s4716671/DRL/MAPPOSE $TMPDIR

tree $TMPDIR

############ RUN CODE:
# Navigate to TMPDIR right directory
cd $TMPDIR/MAPPOSE

# Run training
# TO DO
# python3 -u new_search.py --env cheetah --n_iterations 25

############ SAVING:
# Save results in $TMPDIR/code/2025-hw1-group-12/results to source
# cp -r $TMPDIR/2025-hw2-group-12/src/results /scratch/s4716671/DRL/'Assignment 2'/search_results/pendulum
# TO DO
cp -r $TMPDIR/MAPPOSE/results /scratch/s4716671/MARL/search_results/pendulum