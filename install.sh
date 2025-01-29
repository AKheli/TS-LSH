#!/bin/sh

# Add repository and update package list
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update

# Install system utilities
sudo apt-get install -y git git-lfs curl unzip build-essential

# Git configuration
git config --global credential.helper store
git lfs pull

# Install Python 3.6 and dependencies
sudo apt install -y python3.6 python3.6-distutils python3-distutils python3-apt python3-pip
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1
sudo apt-get install -y python3-pip
pip3 install --upgrade pip

# Install Python packages for Python 3.6
python -m pip install numpy pandas matplotlib sklearn future torch torchvision scipy toml wfdb lshashpy3

# Install TensorFlow and SugarTensor
python -m pip install https://files.pythonhosted.org/packages/86/9f/be0165c6eefd841e6928e54d3d083fa174f92d640fdc52f73a33dc9c54d1/tensorflow-1.4.0-cp36-cp36m-manylinux1_x86_64.whl
python -m pip install sugartensor==1.0.0.2

# Install Python 3.8 and virtual environment tools
sudo apt install -y python3.8 python3.8-venv libpython3.8-dev python3-dev
python3.8 -m venv TSMvenv
. TSLSHvenv/bin/activate
python3.8 -m pip install --upgrade pip

# Install Python packages for Python 3.8
pip3 install saxpy numpy pandas protobuf==3.13.0 tqdm gdown pylab-sdk sqlalchemy matplotlib matplotlib_terminal jsonlines lshashpy3
