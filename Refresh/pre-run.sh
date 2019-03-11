wget http://kinloch.inf.ed.ac.uk/public/Refresh-NAACL18-1-billion-benchmark-wordembeddings.tar.gz
wget http://kinloch.inf.ed.ac.uk/public/Refresh-NAACL18-preprocessed-input-data.tar.gz
wget http://kinloch.inf.ed.ac.uk/public/Refresh-NAACL18-CNN-DM-Filtered-TokenizedSegmented.tar.gz
wget http://kinloch.inf.ed.ac.uk/public/Refresh-NAACL18-baseline-gold-data.tar.gz


tar xvzf Refresh-NAACL18-preprocessed-input-data.tar.gz
tar xvzf Refresh-NAACL18-CNN-DM-Filtered-TokenizedSegmented.tar.gz
tar xvzf Refresh-NAACL18-baseline-gold-data.tar.gz
tar xvzf Refresh-NAACL18-1-billion-benchmark-wordembeddings.tar.gz

# create virtual env
conda create --name referesh python=2.7

# prepare the virtual environment by the followinng command
conda install pip
conda install numpy
conda install -c conda-forge/label/cf201901 tensorflow=0.10
pip install pyrouge

