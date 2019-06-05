#!/bin/bash

Tasks="up ui us hup"

for task in $Tasks; do
        echo $task
        python mtl_coherency.py --seed 290189 --logdir logs --embedding glove --cuda 3 --datadir data/daily_dialog/train --do_train --task $task --model model-3 --learning_rate 1e-3
done
