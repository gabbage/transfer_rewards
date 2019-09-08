#!/bin/bash
#
#SBATCH --job-name=sw_mtl
#SBATCH --output=/ukp-storage-1/buecker/res_stastistics_sw1.txt
#SBATCH --mail-user=sebastian.buecker@stud.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --partition=ukp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

source /ukp-storage-1/buecker/buecker/bin/activate
module purge
module load cuda/9.0

#Tasks="up hup ui us"
Tasks="up hup ui"
Datasets="train validation test"
#Losses="da mtl coin coh"
#Losses="mtl coh"
Losses="mtl coh"

BD=/ukp-storage-1/buecker/transfer_rewards/sub_rewards

for task in $Tasks; do
        for loss in $Losses; do 
                echo $task $loss
                python $BD/mtl_coherency.py --logdir $BD/logs/sw --seed $RANDOM --datadir $BD/data/switchboard --embedding glove --task $task --do_train --do_eval --model model-3 --epochs 10 --num_classes 50 --batch_size 16 --loss $loss --learning_rate 0.0005
        done
done
