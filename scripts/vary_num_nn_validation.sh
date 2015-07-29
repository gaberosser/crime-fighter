#!/bin/bash -l

# set the location
# LOCATION=chicago_northwest
LOCATION=chicago_south

# set the run type
MODULE=scripts.vary_num_nn_validation

# Batch script to run an array job on Legion with the upgraded
# software stack under SGE.

# 1. Force bash
#$ -S /bin/bash

# 2. Request wallclock time (format hours:minutes:seconds).
#$ -l h_rt=3:0:0

# 3. Request RAM.
#$ -l mem=4G

# 4. Request 15 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=10G

# 5. Set up the job array.
#$ -t 1-12

# 6. Set the name of the job.
#$ -N vary_num_nn

# 7. Set the working directory to somewhere in your scratch space.  This is
# a necessary step with the upgraded software stack as compute nodes cannot
# write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/ucesga2/Scratch/crime-fighter/

# 8. Parse parameter file to get variables.
number=$SGE_TASK_ID
paramfile=$HOME/Scratch/crime-fighter/scripts/parameters/vary_num_nn_validation.txt

paramloc=`sed -n ${number}p $paramfile | awk '{print $1}'`
parama=`sed -n ${number}p $paramfile | awk '{print $2}'`
paramb=`sed -n ${number}p $paramfile | awk '{print $3}'`

# 9. Run the program (replace echo with your binary and options).
# cd $HOME
# source .bashrc
cd $HOME/Scratch/crime-fighter
python -m $MODULE $paramloc $parama $paramb
