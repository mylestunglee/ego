#!/bin/bash

for i in `seq 1 10`;
do

	VAR="PSO test RTM $i time"
	FILE="logs/PSO_RTM_$i.txt"
	echo $VAR > $FILE
	echo `./test 3 2 0 $i 20 &>> $FILE`
done

for i in `seq 1 10`;
do

	VAR="Brute force test QUAD $i exec time"
	FILE="logs/BRUTE_QUAD_0001_$i.txt"
	echo $VAR > $FILE
	echo `./test 1 1 0 $i 20 &>> $FILE`
done

for i in `seq 1 10`;
do

	VAR="PSO force test QUAD $i exec time"
	FILE="logs/PSO_QUAD_0001_$i.txt"
	echo $VAR > $FILE
	echo `./test 1 2 0 $i 20 &>> $FILE`
done
