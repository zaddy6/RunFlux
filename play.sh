# !/bin/bash

## Install stuff we need
add-apt-repository -y ppa:rmescandon/yq
apt update
apt install -y screen vim git-lfs unzip yq

# Debug
env

## SETUP ai-toolkit
cd /workspace/ai-toolkit

# Download images 
wget -O images.zip "${IMAGE_ARCHIVE}"
unzip images.zip -d images

# sleep infinity
runpodctl remove pod $RUNPOD_POD_ID
