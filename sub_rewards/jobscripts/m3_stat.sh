#!/bin/bash
#
#SBATCH --job-name=model3_stat
#SBATCH --output=/ukp-storage-1/buecker/res_stastistics.txt
#SBATCH --mail-user=sebastian.buecker@stud.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --partition=ukp
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=4
#SBATCH --mem=14GB
#SBATCH --gres=gpu:1

source /ukp-storage-1/buecker/buecker/bin/activate
module purge
module load cuda/9.0

Tasks="ui us"
Datasets="train validation test"
Losses="da mtl coin coh"

BD=/ukp-storage-1/buecker/transfer_rewards/sub_rewards

for task in $Tasks; do
        for loss in $Losses; do 
                echo $task $loss
                python $BD/mtl_coherency.py --logdir $BD/logs/dd_m3_stat --seed $RANDOM --datadir $BD/data/daily_dialog --embedding glove --task $task --do_train --do_eval --model model-3 --epochs 10 --num_classes 5 --batch_size 128 --loss $loss
        done
done
