#!/bin/bash
#SBATCH --job-name=audit_ev4
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --account=atlas
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
# Run the re-expansion audit on the CURRENT (pre-fix) event 4 parquet. It is
# expected to FAIL on the ordering-dependent checks: that failure is the
# demonstration that the audit catches the inversion Noe will re-expand
# against.
module load python
export PYTHONUNBUFFERED=1
cd /global/cfs/cdirs/atlas/mussonm/cCKF
python3 scripts/audit_expansion.py \
  --parquet /pscratch/sd/m/mussonm/cckf/reexpanded/expanded_event000000004.parquet \
  --trackstates /pscratch/sd/m/mussonm/cckf/modal_backup/results/pilot_1786525888/trackstates_ckf.root \
  --event 4 \
  --json /pscratch/sd/m/mussonm/cckf/results/audit_event4_prefix_v3.json
echo "AUDIT_RC=$?"
