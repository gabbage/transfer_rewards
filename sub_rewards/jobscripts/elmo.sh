#!/bin/bash
#
#SBATCH --job-name=elmo
#SBATCH --output=/ukp-storage-1/buecker/res_elmo.txt
#SBATCH --mail-user=sebastian.buecker@stud.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --partition=ukp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1

source /ukp-storage-1/buecker/buecker/bin/activate
module purge
module load cuda/10.0

Tasks="up" # hup ui us"
Datasets="train validation test"
Losses="mtl"

BD=/ukp-storage-1/buecker/transfer_rewards/sub_rewards

for task in $Tasks; do
        for loss in $Losses; do 
                echo $task $loss
                python $BD/mtl_coherency.py --logdir $BD/logs/dd_model-4 --seed $RANDOM --datadir $BD/data/daily_dialog --embedding elmo --task $task --do_train --do_eval --model elmo-1 --epochs 5 --num_classes 5 --batch_size 5 --loss $loss
        done
done
