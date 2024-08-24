#!/bin/bash

## Install stuff we need
add-apt-repository -y ppa:rmescandon/yq
apt update
apt install -y screen vim git-lfs unzip yq

# Debug
env

## INSTALL ai-toolkit and other stuff
cd /workspace
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
pip install accelerate transformers diffusers huggingface_hub torchvision safetensors lycoris-lora==1.8.3 flatten_json pyyaml oyaml tensorboard kornia invisible-watermark einops toml albumentations pydantic omegaconf k-diffusion open_clip_torch timm prodigyopt controlnet_aux==0.0.7 python-dotenv bitsandbytes hf_transfer lpips pytorch_fid optimum-quanto sentencepiece 

## LOGIN HF
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

## SETUP ai-toolkit
cd /workspace/ai-toolkit

# Download images 
wget -O images.zip "${IMAGE_ARCHIVE}"
unzip images.zip -d images

# Write ai-toolkit config with params passed from Colab notebook
export FOLDER_PATH="/workspace/ai-toolkit/images"
export MODEL_NAME="black-forest-labs/FLUX.1-dev"

cp config/examples/train_lora_flux_24gb.yaml config/${NAME}_train_lora_flux_24gb.yaml

declare -A yaml_params=(
  [config.name]=NAME
  [config.process[0].network.linear]=LORA_RANK
  [config.process[0].network.linear]=LORA_ALPHA
  [config.process[0].trigger_word]=TRIGGER_WORD
  [config.process[0].save.save_every]=SAVE_STEPS
  [config.process[0].datasets[0].folder_path]=FOLDER_PATH
  [config.process[0].train.batch_size]=BATCH_SIZE
  [config.process[0].train.steps]=STEPS
  [config.process[0].train.lr]=LEARNING_RATE
  [config.process[0].sample.seed]=SEED
  [config.process[0].sample.sample_every]=SAMPLE_STEPS
  [config.process[0].model.quantize]=QUANTIZE_MODEL
  [config.process[0].model.name_or_path]=MODEL_NAME
)

for param in "${!yaml_params[@]}"; do
  yq eval ".${param} = env(${yaml_params[$param]})" config/${NAME}_train_lora_flux_24gb.yaml > config/temp.yaml && mv config/temp.yaml config/${NAME}_train_lora_flux_24gb.yaml
done

# Replace the YAML manipulation with a Python script
python3 << EOF
import yaml
import os

# Read the YAML file
with open('images/prompts.yaml', 'r') as file:
    data = yaml.safe_load(file)

# Extract and modify prompts
trigger_word = os.environ['TRIGGER_WORD']
prompts = [f'{trigger_word} style: {prompt}' for prompt in data['prompts']]

# Read the existing config file
config_file = f"config/{os.environ['NAME']}_train_lora_flux_24gb.yaml"
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

# Update the prompts in the config
config['config']['process'][0]['sample']['prompts'] = prompts

# Write the updated config back to the file
with open(config_file, 'w') as file:
    yaml.dump(config, file)
EOF

# upload config
huggingface-cli upload $HF_REPO config/${NAME}_train_lora_flux_24gb.yaml $NAME/config.yaml

## SCHEDULE UPLOADS of samples/adapters every 3 mins 
mkdir -p output/$NAME/samples
touch ${NAME}_ai-toolkit.log

# Function to upload adapter
upload_adapter() {
    while true; do
        echo "Uploading adapter:"
        huggingface-cli upload "$HF_REPO" output/$NAME/${NAME}.safetensors "$NAME/adapters/${NAME}.safetensors"
        sleep 180  # Wait for 3 minutes
    done
}

# Start adapter upload in background
upload_adapter &

huggingface-cli upload $HF_REPO ${NAME}_ai-toolkit.log $NAME/log.txt --every=3 &

# Upload samples every 3 minutes
bash -c 'while true; do huggingface-cli upload $HF_REPO output/$NAME/samples $NAME/samples; sleep 180; done' &

## TRAIN
python run.py config/${NAME}_train_lora_flux_24gb.yaml 2>&1 | tee ${NAME}_ai-toolkit.log

## UPLOAD RESULTS one last time
huggingface-cli upload "$HF_REPO" "output/$NAME/samples" "$NAME/samples"
huggingface-cli upload "$HF_REPO" "output/$NAME/${NAME}.safetensors" "$NAME/adapters/${NAME}.safetensors"
huggingface-cli upload "$HF_REPO" "${NAME}_ai-toolkit.log" "$NAME/log.txt"

# Kill background jobs
kill $(jobs -p)

# sleep infinity
runpodctl remove pod $RUNPOD_POD_ID
