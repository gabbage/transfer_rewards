# Setup #
* install anaconda (my anaconda version:2018.12, python version : 3.7.1.final.0, conda version:4.5.12, conda-build version : 3.17.6) (download it from its website). You can check this information by the following commands:
```
conda list conda$
conda info
```

* create a virtual environment by anaconda 
```
conda env create -f ./documentation/environment-gpu.yml
```

# Run #
If you run the project for the first time, you should update the ```machine``` variable in the ```run.sh``` based on the type of the machine.

In order to run the agent:

```
./run.sh
```

If you want to finish the conversation, type ```bye```

You can check the log files in the directory ```./logs```