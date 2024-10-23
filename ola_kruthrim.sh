#!/bin/bash

# Create Miniconda directory and download the installer
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh

# Install Miniconda
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3

# Clean up the installer
rm ~/miniconda3/miniconda.sh

# Initialize conda for bash and zsh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

# Source .bashrc
source ~/.bashrc

# Update package lists
sudo apt-get update -y

# Install virtualenv
pip install virtualenv

# Create and activate a new conda environment
conda create -n venv python=3.8 -y
conda activate venv

# Check Python version
python --version

# Install python3-pip
sudo apt-get update -y
sudo apt install python3-pip -y

# Install pip using get-pip.py
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.8 get-pip.py

# Check pip version
pip --version

# Clone the GitHub repository
git clone -b Visual_Prompt https://github.com/KBaneesh1/ELMPMKG

# Navigate to the cloned repository
cd ELMPMKG/

# Install the required packages
python3.8 -m pip install -r requirements.txt

# Install gdown
pip install gdown

# Download files using gdown
gdown --id 17eJ2qfxCBCZGQOnbtYo_uhbckI2DExOm
python3.8 exec.py
gdown --id 18WL6twcgiReCXXkRVAo7zK-FAua8B5IG

# Install a specific version of protobuf
pip install protobuf==3.20.*

# Install PyTorch and its dependencies
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

# Run the pretraining script
bash scripts/pretrain_fb15k-237-image.sh
