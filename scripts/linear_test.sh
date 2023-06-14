#!/bin/bash

for i in 1.0 1.6

do

python ../grasp_moving.py  --speed $i --traj "circle" --predict "True"

python ../grasp_moving.py  --speed $i --traj "circle" --predict "False"

done
