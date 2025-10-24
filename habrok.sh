#!/bin/bash
#SBATCH --job-name=mappose_initial_run         # Job name
#SBATCH --output=mappose-initial-run-%j.log
#SBATCH --nodes=1                     # Number of nodes (use 1 node)
#SBATCH --ntasks=1                    # One task
#SBATCH --gpus-per-node=a100:1              
#SBATCH --mem=8GB                     # Total memory for the job (adjust based on need)
#SBATCH --time=16:00:00              # Time limit for the job 

# remove all previously loaded modules
module purge

# load python 3.8.16
module load Python/3.11.5-GCCcore-13.2.0  
 
# activate virtual environment
source $HOME/venvs/mappose/bin/activate

mkdir -p /scratch/s4716671/MARL/run_results/$SLURM_JOB_ID

############ GETTING THE CODE
mkdir -p $TMPDIR

# copy code into TMPDIR
cp -r /scratch/s4716671/MARL/MAPPOSE $TMPDIR

tree $TMPDIR

############ RUN CODE:
# Navigate to TMPDIR right directory
cd $TMPDIR/MAPPOSE

# Run training
python3 -u testing_out.py

############ SAVING:
# Save results
cp -r $TMPDIR/MAPPOSE/results /scratch/s4716671/MARL/run_results/$SLURM_JOB_ID