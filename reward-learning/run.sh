#!/bin/bash

###
##
# define variables
machine='local' # possible values: {'local', 'gpu'}, where 'local' has no GPU and cuda

experiment='online'
 
log_file='./logs/'$experiment'.log'

env='transfer_reward'

modelpath='./models/regression.py'
dataset='duc04'
loss='ce'

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
if [[ $machine == 'gpu' ]]
then
	git checkout logs/*
	git checkout models/*
	git pull

fi


###
##
# Finally execute the model
if [[ $machine == 'local' ]]
then
	pythonw run.py -m $modelpath -d $dataset -l $loss > $log_file
else
	python run.py -m $modelpath -d $dataset -l $loss > $log_file
fi