#!/bin/bash

#------- Job Description -------

#SBATCH --job-name='fingerprint creation'
#SBATCH --comment='fingerprint for tanimoto comparison'

#------- Parametrization -------

#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=bqhn01
#SBATCH --time=10-0:0:0

#------- Input/Output -------

#SBATCH --output="/home/roberto_olayo/mole_antimicrobial_potential/workflow/logdir/out/%x-%j.out"
#SBATCH --error="/home/roberto_olayo/mole_antimicrobial_potential/workflow/logdir/out/%x-%j.err"

#------- Command -------
python 07.pubchem_exploration_fingerprints.py