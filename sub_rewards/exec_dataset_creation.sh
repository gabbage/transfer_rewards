#!/bin/bash

Tasks="up hup ui us"
Datasets="train validation test"

for dset in $Datasets; do
        for task in $Tasks; do
                echo $dset $task
                python create_coherency_dataset.py --seed 290189 --amount 20 --datadir data/daily_dialog/$dset --task $task
        done
done
