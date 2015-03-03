#!/bin/bash -l

# Batch script to run an array job on Legion with the upgraded
# software stack under SGE.

# 1. Force bash
#$ -S /bin/bash

# 2. Request wallclock time (format hours:minutes:seconds).
#$ -l h_rt=1:0:0

# 3. Request RAM.
#$ -l mem=2G

# 4. Request 10 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=10G

# 5. Set up the job array.
#$ -t 1-100

# 6. Set the name of the job.
#$ -N vary_min_bandwidths_trigger_only

# 7. Set the working directory to somewhere in your scratch space.  This is
# a necessary step with the upgraded software stack as compute nodes cannot
# write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/ucesga2/Scratch/crime-fighter/

# 8. Parse parameter file to get variables.
number=$SGE_TASK_ID
paramfile=/home/ucesga2/Scratch/crime-fighter/scripts/vary_min_bandwidth_params.txt

crime_type=`sed -n ${number}p $paramfile | awk '{print $1}'`
min_t_bd=`sed -n ${number}p $paramfile | awk '{print $2}'`
min_d_bd=`sed -n ${number}p $paramfile | awk '{print $3}'`

# 9. Run the program (replace echo with your binary and options).
source $HOME/scripts/python_modules.sh
cd $HOME/Scratch/crime-fighter
python -m scripts.vary_min_bandwidths_trigger_only $crime_type $min_t_bd $min_d_bd
