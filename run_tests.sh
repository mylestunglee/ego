#!/bin/bash

for i in `seq 1 10`;
do

	VAR="PSO test RTM $i time"
	FILE="PSO_RTM_$i.txt"
	echo $VAR > $FILE
	echo `./test 3 2 0 &>> $FILE`
done

for i in `seq 1 10`;
do

	VAR="Brute force test QUAD $i exec time"
	FILE="BRUTE_QUAD_0001_$i.txt"
	echo $VAR > $FILE
	echo `./test 1 1 0 &>> $FILE`
done

for i in `seq 1 10`;
do

	VAR="PSO force test QUAD $i exec time"
	FILE="PSO_QUAD_0001_$i.txt"
	echo $VAR > $FILE
	echo `./test 1 2 0 &>> $FILE`
done
