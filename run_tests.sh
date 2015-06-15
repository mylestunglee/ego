#!/bin/bash
NAME="NORMALISE_LOG_COST"

#for j in 6 
#do
#  for f in 1
#  do
#      N_SIMS=50
#      VAR="Search = $f test QUAD $NAME lambda = $j n_sims = $N_SIMS"
#      FILE="logs/"
#      FILE+="$NAME"
#      FILE+="_$f" 
#      FILE+="_QUAD" 
#      FILE+="_"
#      FILE+="$j"
#      FILE+="_"
#      FILE+="$N_SIMS"
#      FILE+=".txt"
#      echo $VAR > $FILE
#      echo `./test 1 $f 1 $j $N_SIMS 1 &>> $FILE`
#  done
#done

for j in 2
do
  for f in 3
  do
      VAR="Search = $f test RTM $NAME lambda = $j n_sims = $N_SIMS"
      FILE="logs/"
      FILE+="$NAME"
      FILE+="_$f" 
      FILE+="_RTM" 
      FILE+="_"
      FILE+="$j"
      FILE+="_"
      FILE+="$N_SIMS"
      FILE+=".txt"
      echo $VAR > $FILE;
      echo `./test 3 $f 1 $j $N_SIMS 1 &>> $FILE`
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
