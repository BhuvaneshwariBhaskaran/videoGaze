#!/bin/bash

today=`date '+%Y_%m_%d__%H_%M_%S'`;
# today='8vids'
echo $today

mv shot/output shot/output_$today
mkdir shot/output
echo 'Renamed to shot/output_'$today

mv detect/output detect/output_$today
mkdir detect/output
echo 'Renamed to detect/output_'$today

mv track/output track/output_$today
mkdir track/output
echo 'Renamed to track/output_'$today

mv crop/output crop/output_$today
mkdir crop/output
echo 'Renamed to crop/output_'$today


mv extract/output extract/output_$today
mkdir extract/output
echo 'Renamed to extract/output_'$today


# mv videoout/output videoout/output_$today
# mkdir videoout/output
# echo 'Renamed to videoout/output_'$today

