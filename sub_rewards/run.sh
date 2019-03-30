#!/bin/bash

###
##
# define variables
machine='gpu' # possible values: {'local', 'gpu'}, where 'local' has no GPU and cuda

env='transfer_reward' 

experiment='dialog_act_classifier_bow'
 
log_file='./logs/'$experiment



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
# dialog_act_classifier
ipython model/dialog_act_classifier.py &> $log_file

