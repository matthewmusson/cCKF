#!/bin/bash
#SBATCH --job-name=modfail
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=atlas
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
module load python
S=$SCRATCH/cckf
mkdir -p $S/modfail_v1
for E in $(seq 0 31); do
  echo $E
done | xargs -P 8 -I{} python3 $S/modfail_analysis.py {} $S
echo "ALL_DONE $(ls $S/modfail_v1/*.npz 2>/dev/null | wc -l)/32"
