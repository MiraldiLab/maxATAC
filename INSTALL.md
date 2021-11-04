# maxATAC: a suite of user-friendly, deep neural network models for transcription factor binding prediction from ATAC-seq

## Install Instructions

To install maxATAC begin by cloning the code base
1. git clone https://github.com/MiraldiLab/maxATAC.git

cd maxATAC
After moving into the cloned directory, please run the following command:

2. git clone --recurse-submodules # this will clone all the recursive directories that are needed to run maxATAC

maxATAC can be run using an environment or using docker. To install using an environment, begin by creating an environment on your computer

# Create your env
3. conda create -n my_env python=3.9 or python3.9 -m venv env_name

activate your environment and install maxatac:

4. pip3 install -r requirements.txt #try to do --constraint instead of -r

5. pip3 install -e . # pip3 install . does not work # TO DO


