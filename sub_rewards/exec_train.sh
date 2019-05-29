#!/bin/bash

Tasks="up hup ui us"

for task in $Tasks; do
        echo $task
        python mtl_coherency.py --seed 290189 --logdir logs --embedding glove --cuda 0 --datadir data/daily_dialog/test/ --do_train --task $task
done
