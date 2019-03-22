#!/bin/bash

###
##
# define variables
machine='gpu' # possible values: {'local', 'gpu'}, where 'local' has no GPU and cuda

env='rl-convagent' 

experiment='movie-dialogue-rl'
 
log_file='./logs/'$experiment

###
##
# check if enough argument exist
if [[ "$1" != "train" ]]  && [[ "$1" != "test" ]]
then 
	echo 'the first argument should be the mode of the system: train|test ?'
	exit
else
	mode=$1
fi

if [[ "$2" != "seq2seq" ]]  && [[ "$2" != "drl" ]]
then 
	echo 'the second argument should be the model used in the system: seq2seq|drl ?'
	exit
else
	model=$2
	log_file='./logs/'$experiment-$mode-$model'.log'
fi

if [[ $mode == "test" ]]
then 
	if  [[ "$3" == "" ]] || [[ "$4" == "" ]] || [[ "$5" == "" ]] 
	then 
		echo 'in test mode, we need to have the path to the model, the input file, and the output file'
		exit
	else
		PATH_TO_MODEL=$3
		INPUT_FILE=$4
		OUTPUT_FILE=$5
		GOLD_FILE=$6

		PATH_TO_MODEL_BASE=`basename $PATH_TO_MODEL`
		INPUT_FILE_BASE=`basename $INPUT_FILE`
		OUTPUT_FILE_BASE=`basename $OUTPUT_FILE`
		GOLD_FILE_BASE=`basename $GOLD_FILE`
		log_file='./logs/'$experiment-$mode-$model-path:$PATH_TO_MODEL_BASE-inp:$INPUT_FILE_BASE-out:$OUTPUT_FILE_BASE-gold:$GOLD_FILE_BASE'.log'
	fi
fi


###
##
#check if we are in RL virtual environment
if [[ $VIRTUAL_ENV != $env ]]
then

    source activate $env


	###
	##
	# export the spec of the running environment (rl)
	conda list > ./documentation/environment-packages-$machine.txt

	conda list --explicit > ./documentation/environment-info-$machine.txt 

	conda env export > ./documentation/environment-$machine.yml

	conda env export --no-builds > ./documentation/environment-nobuilds-$machine.yml


fi


###
##
# run the latest version of the model by getting it from GitHub
if [[ $machine == "gpu" ]]
then
	git checkout logs/*
	git checkout models/*
	git pull

fi


###
##
# Finally execute the model
if [[ $mode == "train" ]]
then

	if [[ $model == "seq2seq" ]]
	then
		 CUDA_VISIBLE_DEVICES=0 python python/train.py &> $log_file

	elif [[ $model == "drl" ]]
	then
		CUDA_VISIBLE_DEVICES=0 python python/RL/train.py &> $log_file
	fi

elif [[ $mode == "test" ]]
then
	if [[ $model == "seq2seq" ]]
	then
		
		CUDA_VISIBLE_DEVICES=0 python python/test.py $PATH_TO_MODEL $INPUT_FILE $OUTPUT_FILE &> $log_file

		CUDA_VISIBLE_DEVICES=0 python python/metrics.py $OUTPUT_FILE  $GOLD_FILE &>> $log_file
	fi
fi

