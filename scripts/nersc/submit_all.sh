#!/bin/bash
declare -A PILOT=( [0]=pilot_1786524194 [2]=pilot_1786524971 [3]=pilot_1786524971
 [4]=pilot_1786525888 [5]=pilot_1786525888 [6]=pilot_1786527639 [7]=pilot_1786527639
 [8]=pilot_1786529841 [9]=pilot_1786529841 [10]=pilot_1786531084 [11]=pilot_1786531084
 [12]=pilot_1786532999 [13]=pilot_1786532999 [14]=pilot_1786534329 [15]=pilot_1786534329
 [16]=pilot_1786536577 [17]=pilot_1786536577 [18]=pilot_1786538022 [19]=pilot_1786538022
 [20]=pilot_1786539360 [21]=pilot_1786539360 [22]=pilot_1786540365 [23]=pilot_1786540365
 [24]=pilot_1786541815 [25]=pilot_1786541815 [26]=pilot_1786542563 [27]=pilot_1786542563
 [28]=pilot_1786543789 [29]=pilot_1786543789 [30]=pilot_1786547065 [31]=pilot_1786547065 )
D=$SCRATCH/cckf/modal_backup/results; O=$SCRATCH/cckf/reexpanded; n=0
for ev in 0 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31; do
  case $ev in 5|7|14|17|18) NCH=8; T=02:45:00;; 0|2|3) NCH=4; T=02:30:00;; *) NCH=2; T=01:30:00;; esac
  jid=$(sbatch --parsable --qos=regular --time=$T --job-name=rxe${ev} \
    --output=$SCRATCH/cckf/logs/rxe_ev${ev}_%j.out \
    $SCRATCH/cckf/reexpand_event.sbatch $D/${PILOT[$ev]} $ev $O/expanded_event$(printf "%09d" $ev).parquet $NCH) \
    && n=$((n+1)) || echo "SUBMIT_FAIL ev=$ev"
done
echo "SUBMITTED=$n"
