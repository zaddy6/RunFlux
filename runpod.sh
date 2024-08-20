# !/bin/bash

# apt update
# apt install -y screen vim git-lfs

# Install stuff we need
add-apt-repository -y ppa:rmescandon/yq
apt update
apt install -y screen vim git-lfs unzip yq

cd /workspace

echo ENV
env

# INSTALL ai-toolkit
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
# pip install torch
# pip install -r requirements.txt
# pip install -U accelerate transformers diffusers huggingface_hub #Optional, run it if you run into issues
pip install accelerate transformers diffusers huggingface_hub torchvision safetensors lycoris-lora==1.8.3 flatten_json pyyaml oyaml tensorboard kornia invisible-watermark einops toml albumentations pydantic omegaconf k-diffusion open_clip_torch timm prodigyopt controlnet_aux==0.0.7 python-dotenv bitsandbytes hf_transfer lpips pytorch_fid optimum-quanto sentencepiece 

# SETUPai-toolkit

# Download images folder 
# export IMAGE_ARCHIVE="https://drive.google.com/uc?export=download&id=1D7vCHhbSttZ-YAm1uABocsevDQT-fxd0"
cd /workspace/ai-toolkit
wget -O images.zip "${IMAGE_ARCHIVE}"
mkdir -p images
cd images
unzip ../images.zip

# Setup ostris/ai-toolkit config
# export LORA_RANK=16
# export LORA_ALPHA=16
# export SAVE_STEPS=250
# export BATCH_SIZE=1
# export STEPS=10
# export LEARNING_RATE=0.0001
# export SEED=42
# export QUANTIZE_MODEL=true
# export TRIGGER_WORD="p3r5on"
export FOLDER_PATH="/workspace/ai-toolkit/images"
export MODEL_NAME="black-forest-labs/FLUX.1-dev"

cd /workspace/ai-toolkit/config
cp examples/train_lora_flux_24gb.yaml .

declare -A yaml_params=(
  [config.process[0].network.linear]=LORA_RANK
  [config.process[0].network.linear]=LORA_ALPHA
  [config.process[0].trigger_word]=TRIGGER_WORD
  [config.process[0].save.save_every]=SAVE_STEPS
  [config.process[0].datasets[0].folder_path]=FOLDER_PATH
  [config.process[0].train.batch_size]=BATCH_SIZE
  [config.process[0].train.steps]=STEPS
  [config.process[0].train.lr]=LEARNING_RATE
  [config.process[0].sample.seed]=SEED
  [config.process[0].model.quantize]=QUANTIZE_MODEL
  [config.process[0].model.name_or_path]=MODEL_NAME
)

for param in "${!yaml_params[@]}"; do
  yq eval ".${param} = env(${yaml_params[$param]})" train_lora_flux_24gb.yaml > temp.yaml && mv temp.yaml train_lora_flux_24gb.yaml
done

# TRAIN
# log in to HF
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

# start ai-toolkit
cd /workspace/ai-toolkit
python run.py config/train_lora_flux_24gb.yaml

# UPLOAD RESULT

# export HF_REPO="g-ronimo/FLUX1-dev-LoRA"
cd /workspace/ai-toolkit/output/my_first_flux_lora_v1
huggingface-cli upload $HF_REPO config.yaml 
huggingface-cli upload $HF_REPO samples samples
huggingface-cli upload $HF_REPO *.safetensors

# sleep infinity

runpodctl remove pod $RUNPOD_POD_ID

