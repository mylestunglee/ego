#!/bin/bash
NAME="_run_2"
for j in `seq 1 6`
do
  for f in `seq 1 2`
  do
    for k in `seq 0 2`
    do
      N_SIMS=$((5 * 10**$k))
      VAR="Search = $f test RTM $NAME lambda = $j n_sims = $N_SIMS"
      FILE="logs/Search_$f" 
      FILE+="_RTM_" 
      FILE+="$NAME"
      FILE+="_"
      FILE+="$j"
      FILE+="_"
      FILE+="$N_SIMS"
      FILE+=".txt"
      echo $VAR > $FILE;
      echo `./test 3 $f 0 $j $N_SIMS &>> $FILE`
    done
  done
done

for j in `seq 1 6`
do
  for f in `seq 1 2`
  do
    for k in `seq 0 2`
    do
      N_SIMS=$((5 * 10**$k))
      VAR="Search = $f test QUAD $NAME lambda = $j n_sims = $N_SIMS"
      FILE="logs/Search_$f" 
      FILE+="_QUAD_" 
      FILE+="$NAME"
      FILE+="_"
      FILE+="$j"
      FILE+="_"
      FILE+="$N_SIMS"
      FILE+=".txt"
      echo $VAR > $FILE
      echo `./test 1 $f 0 $j $N_SIMS &>> $FILE`
    done
  done
done

#for i in `seq 1 10`;
#do
#
#	VAR="PSO force test QUAD $i exec time"
#	FILE="logs/PSO_QUAD_0001_$i.txt"
#	echo $VAR > $FILE
#	echo `./test 1 2 0 $i 20 &>> $FILE`
#done
