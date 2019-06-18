#!/bin/bash

Tasks="up ui us hup"

for task in $Tasks; do
        echo $task
	echo "\n"
        python mtl_coherency.py --logdir logs --embedding glove --cuda 2 --datadir data/daily_dialog/train --do_train --task $task --model model-3
done
