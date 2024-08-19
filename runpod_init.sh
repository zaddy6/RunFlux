# !/bin/bash

# install screen ..
apt update
apt install -y screen vim git-lfs

# .. continue in screen
screen -dm /workspace/runpod.sh
