#!/bin/bash

Tasks="up hup ui us"
Datasets="train validation test"

for task in $Tasks; do
	for dset in $Datasets; do
                echo $dset $task
                python mtl_coherency.py --seed 290189 --datadir data/daily_dialog/$dset --embedding glove --task $task --do_eval --model model-3 --cuda 1
        done
done
