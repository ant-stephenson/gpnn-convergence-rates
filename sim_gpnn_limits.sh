#!/bin/bash

#SBATCH --job-name=sim_gpnn_limits
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=15G
#SBATCH --array=1-160
## SBATCH --array=0
#SBATCH --mail-type=END,FAIL

SCRIPT_DIR=/user/work/...

module load lang/python/anaconda/3.8.5-2021-AM

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/lang/anaconda.3.8.5-2021-AM/lib

source $SCRIPT_DIR/projenv/bin/activate

echo $SLURM_JOB_START_TIME

DIM=50
LS=0.5
DESCRIPTION="moren_misspec"
DIR="sim_gpnn_results/${SLURM_ARRAY_JOB_ID}"
INFILE=None #DIM{$DIM}_LENSCALE{$LS}_{1} #None

if [ ! -d $DIR ]; then
  mkdir $DIR
fi

OUTFILE="${DIR}/sim_gpnn_limits_results_${DESCRIPTION}_${SLURM_ARRAY_JOB_ID}"
if [ $SLURM_ARRAY_TASK_ID != 0 ]; then
  OUTFILE="${OUTFILE}_${SLURM_ARRAY_TASK_ID}"
fi

NSEED=5

# only if array task id = 0, otherwise remove N_TRAIN loop
for N_TRAIN in 1000
  do
  for SEED in $(seq 1 $NSEED)
    do
      python3 $SCRIPT_DIR/gpprediction/simulate.py \
          -n_train $N_TRAIN -n_test 1000 -d $DIM \
          -tker Exp -tks 0.9 -tl $LS -tnv 0.1 -aker "RBF" "Exp" -aks 0.8 -al 0.75 -anv 0.2 \
          -varpar "lengthscale" -numvals 40 -numnn 400 -seed $SEED \
          -out "${OUTFILE}" \
          -array_idx $SLURM_ARRAY_TASK_ID \
      
  done
done

deactivate
