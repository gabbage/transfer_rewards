#!/bin/bash

Tasks="up hup ui us"
Datasets="validation test"

for dset in $Datasets; do
        for task in $Tasks; do
                echo $dset $task
                python mtl_coherency.py --seed 290189 --datadir data/daily_dialog/$dset --embedding glove --task $task --do_eval
        done
done