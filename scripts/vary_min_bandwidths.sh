#!/bin/bash -l

# Batch script to run an array job on Legion with the upgraded
# software stack under SGE.

# 1. Force bash
#$ -S /bin/bash

# 2. Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=0:0:3

# 3. Request 1 gigabyte of RAM.
#$ -l mem=512M

# 4. Request 15 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=10G

# 5. Set up the job array.  In this instance we have requested 1000 tasks
# numbered 1 to 1000.
#$ -t 1-25

# 6. Set the name of the job.
#$ -N array-params

# 7. Set the working directory to somewhere in your scratch space.  This is
# a necessary step with the upgraded software stack as compute nodes cannot
# write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/ucesga2/Scratch/crime_fighter/

# 8. Parse parameter file to get variables.
number=$SGE_TASK_ID
paramfile=/home/ucesga2/Scratch/crime-fighter/scripts/vary_min_bandwidth_params.txt

index=`sed -n ${number}p $paramfile | awk '{print $1}'`
variable1=`sed -n ${number}p $paramfile | awk '{print $2}'`
variable2=`sed -n ${number}p $paramfile | awk '{print $3}'`
variable3=`sed -n ${number}p $paramfile | awk '{print $4}'`

# 9. Run the program (replace echo with your binary and options).

echo $index $variable1 $variable2 $variable3