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
pip install accelerate transformers diffusers huggingface_hub torchvision safetensors lycoris-lora==1.8.3 flatten_json pyyaml oyaml tensorboard kornia invisible-watermark einops toml albumentations pydantic omegaconf k-diffusion open_clip_torch timm prodigyopt controlnet_aux==0.0.7 python-dotenv bitsandbytes hf_transfer lpips pytorch_fid optimum-quanto sentencepiece PyYAML

## LOGIN HF
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

## SETUP ai-toolkit
cd /workspace/ai-toolkit

# Download images 
wget -O images.zip "${IMAGE_ARCHIVE}"
unzip images.zip -d images

# Replace the YAML modification section with Python script
python3 << END
import yaml
import os

# Load the YAML file
config_file = f"config/{os.environ['NAME']}_train_lora_flux_24gb.yaml"
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

# Update the configuration
yaml_params = {
    'config.name': 'NAME',
    'config.process[0].network.linear': 'LORA_RANK',
    'config.process[0].network.alpha': 'LORA_ALPHA',
    'config.process[0].trigger_word': 'TRIGGER_WORD',
    'config.process[0].save.save_every': 'SAVE_STEPS',
    'config.process[0].datasets[0].folder_path': 'FOLDER_PATH',
    'config.process[0].train.batch_size': 'BATCH_SIZE',
    'config.process[0].train.steps': 'STEPS',
    'config.process[0].train.lr': 'LEARNING_RATE',
    'config.process[0].sample.seed': 'SEED',
    'config.process[0].sample.sample_every': 'SAMPLE_STEPS',
    'config.process[0].model.quantize': 'QUANTIZE_MODEL',
    'config.process[0].model.name_or_path': 'MODEL_NAME'
}

for yaml_path, env_var in yaml_params.items():
    keys = yaml_path.split('.')
    current = config
    for key in keys[:-1]:
        if key.endswith(']'):
            key, index = key[:-1].split('[')
            current = current[key][int(index)]
        else:
            current = current[key]
    current[keys[-1]] = os.environ[env_var]

# Load prompts from the YAML file
with open('images/prompts.yaml', 'r') as file:
    prompts_data = yaml.safe_load(file)

# Modify prompts
modified_prompts = [f'[trigger] style: {prompt}' for prompt in prompts_data['prompts']]

# Update the prompts in the config
config['config']['process'][0]['sample']['prompts'] = modified_prompts

# Save the updated configuration
with open(config_file, 'w') as file:
    yaml.dump(config, file)
END

# upload config
huggingface-cli upload $HF_REPO config/${NAME}_train_lora_flux_24gb.yaml

## SCHEDULE UPLOADS of samples/adapters every 3 mins 
mkdir -p output/$NAME/samples
touch ${NAME}_ai-toolkit.log

huggingface-cli upload $HF_REPO output/$NAME --include="*.safetensors" --every=3 &
huggingface-cli upload $HF_REPO ${NAME}_ai-toolkit.log --every=3 &

# (for some reason --every does not upload with samples/ dir, no error, no idea -> bash loop)
bash -c 'while true; do huggingface-cli upload $HF_REPO output/$NAME/samples samples; sleep 180; done' &

## TRAIN
python run.py config/${NAME}_train_lora_flux_24gb.yaml 2>&1 | tee ${NAME}_ai-toolkit.log

## UPLOAD RESULTS one last time
huggingface-cli upload $HF_REPO output/$NAME/samples ${NAME}_samples
huggingface-cli upload $HF_REPO output/$NAME --include="*.safetensors"
huggingface-cli upload $HF_REPO ${NAME}_ai-toolkit.log

# sleep infinity
runpodctl remove pod $RUNPOD_POD_ID
