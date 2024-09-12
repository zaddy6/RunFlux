#!/bin/bash

# Clone the repository
cd /workspace
git clone https://github.com/zaddy6/RunFlux.git

# Execute play.sh
bash /workspace/RunFlux/play.sh

# Debug
env

# Change directory to ai-toolkit
cd /workspace/ai-toolkit
ls

# Infinite loop to keep the script running
while true; do
    sleep 60  # Sleep for 60 seconds before looping
done
