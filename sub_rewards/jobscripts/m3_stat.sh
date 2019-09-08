#!/bin/bash
#
#SBATCH --job-name=dd_model3_stat
#SBATCH --output=/ukp-storage-1/buecker/res_stastistics3.txt
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

Tasks="us"
Datasets="train validation test"
#Losses="da mtl coin coh"
Losses="coh mtl"

BD=/ukp-storage-1/buecker/transfer_rewards/sub_rewards

for task in $Tasks; do
        for loss in $Losses; do 
                echo $task $loss
                python $BD/mtl_coherency.py --logdir $BD/logs/dd_m3_stat --seed $RANDOM --datadir $BD/data/daily_dialog --embedding glove --task $task --do_train --do_eval --model model-3 --epochs 20 --num_classes 5 --batch_size 128 --loss $loss
        done
done
