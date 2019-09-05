#!/bin/bash

#Tasks="up hup ui us"
#Datasets="train validation test"

#for task in $Tasks; do
        #for task2 in $Tasks; do
                ##python mtl_coherency.py --seed 290189 --datadir data/daily_dialog/$dset --embedding glove --task $task --do_eval --model model-3 --cuda 1
                #ootm="data/daily_dialog/model-3_task-${task2}_loss-mtl_epoch-19.ckpt"
                #python mtl_coherency.py --logdir logs/oot --seed $RANDOM --datadir data/daily_dialog --embedding glove --task $task --do_eval --model model-3 --cuda 1 --oot_model $ootm
        #done
#done


Tasks="up hup ui us"
Datasets="train validation test"
Losses="coh mtl"

#BD=/ukp-storage-1/buecker/transfer_rewards/sub_rewards
BD=/home/buecker/transfer_rewards/sub_rewards
BD=.

for task in $Tasks; do
        for loss in $Losses; do 
                echo $task $loss
                python $BD/mtl_coherency.py --logdir $BD/logs/cosine --seed $RANDOM --datadir $BD/data/daily_dialog --embedding glove --task $task --do_eval --model cosine --num_classes 5
        done
done
