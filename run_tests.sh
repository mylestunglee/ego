#!/bin/bash



#for j in 6
#do
#  for f in 1
#  do
#      N_SIMS=100
#      VAR="Search = $f test QUAD $NAME lambda = $j n_sims = $N_SIMS"
#      NAME="random_raw_log_"
#      FILE="quad_logs/"
#      FILE+="$NAME"
#      FILE+="$f" 
#      FILE+="_"
#      FILE+="$j"
#      FILE+="_"
#      FILE+="$N_SIMS"
#      FILE+="_run3.txt"
#      echo $VAR > $FILE
#      echo `./test 1 $f 0 $j $N_SIMS 1 &>> $FILE`
#  done
#done

for j in 6
do
  for f in 3
  do
      N_SIMS=100
      NAME="random_raw_log_"
      VAR="Search = $f test RTM $NAME lambda = $j n_sims = $N_SIMS"
      FILE="rtm_logs/"
      FILE+="$NAME"
      FILE+="$f" 
      FILE+="_"
      FILE+="$j"
      FILE+="_"
      FILE+="$N_SIMS"
      FILE+="_run2.txt"
      echo $VAR > $FILE;
      echo `./test 3 $f 0 $j $N_SIMS 1 &>> $FILE`
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
