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
export LORA_RANK=16
export LORA_ALPHA=16
export SAVE_STEPS=250
export FOLDER_PATH="/workspace/ai-toolkit/images"
export BATCH_SIZE=1
export STEPS=4000
export LEARNING_RATE=0.0001
export SEED=42
export QUANTIZE_MODEL=true
export MODEL_NAME="black-forest-labs/FLUX.1-dev"

cd /workspace/ai-toolkit/config
cp examples/train_lora_flux_24gb.yaml .

declare -A yaml_params=(
  [config.process[0].network.linear]=LORA_RANK
  [config.process[0].network.linear_alpha]=LORA_ALPHA
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
cd /workspace/ai-toolkit

# python run.py config/train_lora_flux_24gb.yml

echo TRAINING DONE

# UPLOAD RESULT

# log in to HF
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

# export HF_REPO="g-ronimo/FLUX1-dev-LoRA"
huggingface-cli upload $HF_REPO /workspace/ai-toolkit/config/train_lora_flux_24gb.yaml
huggingface-cli upload $HF_REPO /workspace/ai-toolkit/output/*.safetensors

# sleep infinity

runpodctl remove pod $RUNPOD_POD_ID

# if [ "$DEBUG" == "True" ]; then
#     echo "Launch LLM AutoEval in debug mode"
# fi

# # Run evaluation
# if [ "$BENCHMARK" == "nous" ]; then
#     git clone -b add-agieval https://github.com/dmahan93/lm-evaluation-harness
#     cd lm-evaluation-harness
#     pip install -e .

#     benchmark="agieval"
#     echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [1/4] =================="
#     python main.py \
#         --model hf-causal \
#         --model_args pretrained=$MODEL_ID,trust_remote_code=$TRUST_REMOTE_CODE,dtype=float16 \
#         --tasks agieval_aqua_rat,agieval_logiqa_en,agieval_lsat_ar,agieval_lsat_lr,agieval_lsat_rc,agieval_sat_en,agieval_sat_en_without_passage,agieval_sat_math \
#         --device cuda:$cuda_devices \
#         --batch_size auto \
#         --output_path ./${benchmark}.json

#     benchmark="gpt4all"
#     echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [2/4] =================="
#     python main.py \
#         --model hf-causal \
#         --model_args pretrained=$MODEL_ID,trust_remote_code=$TRUST_REMOTE_CODE,dtype=float16 \
#         --tasks hellaswag,openbookqa,winogrande,arc_easy,arc_challenge,boolq,piqa \
#         --device cuda:$cuda_devices \
#         --batch_size auto \
#         --output_path ./${benchmark}.json

#     benchmark="truthfulqa"
#     echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [3/4] =================="
#     python main.py \
#         --model hf-causal \
#         --model_args pretrained=$MODEL_ID,trust_remote_code=$TRUST_REMOTE_CODE,dtype=float16 \
#         --tasks truthfulqa_mc \
#         --device cuda:$cuda_devices \
#         --batch_size auto \
#         --output_path ./${benchmark}.json

#     benchmark="bigbench"
#     echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [4/4] =================="
#     python main.py \
#         --model hf-causal \
#         --model_args pretrained=$MODEL_ID,trust_remote_code=$TRUST_REMOTE_CODE,dtype=float16 \
#         --tasks bigbench_causal_judgement,bigbench_date_understanding,bigbench_disambiguation_qa,bigbench_geometric_shapes,bigbench_logical_deduction_five_objects,bigbench_logical_deduction_seven_objects,bigbench_logical_deduction_three_objects,bigbench_movie_recommendation,bigbench_navigate,bigbench_reasoning_about_colored_objects,bigbench_ruin_names,bigbench_salient_translation_error_detection,bigbench_snarks,bigbench_sports_understanding,bigbench_temporal_sequences,bigbench_tracking_shuffled_objects_five_objects,bigbench_tracking_shuffled_objects_seven_objects,bigbench_tracking_shuffled_objects_three_objects \
#         --device cuda:$cuda_devices \
#         --batch_size auto \
#         --output_path ./${benchmark}.json

#     end=$(date +%s)
#     echo "Elapsed Time: $(($end-$start)) seconds"
    
#     python ../llm-autoeval/main.py . $(($end-$start))

# elif [ "$BENCHMARK" == "openllm" ]; then
#     git clone https://github.com/EleutherAI/lm-evaluation-harness
#     cd lm-evaluation-harness
#     pip install -e .
#     pip install accelerate

#     benchmark="arc"
#     echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [1/6] =================="
#     accelerate launch -m lm_eval \
#         --model hf \
#         --model_args pretrained=${MODEL_ID},dtype=auto,trust_remote_code=$TRUST_REMOTE_CODE \
#         --tasks arc_challenge \
#         --num_fewshot 25 \
#         --batch_size auto \
#         --output_path ./${benchmark}.json

#     benchmark="hellaswag"
#     echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [2/6] =================="
#     accelerate launch -m lm_eval \
#         --model hf \
#         --model_args pretrained=${MODEL_ID},dtype=auto,trust_remote_code=$TRUST_REMOTE_CODE \
#         --tasks hellaswag \
#         --num_fewshot 10 \
#         --batch_size auto \
#         --output_path ./${benchmark}.json

#     benchmark="mmlu"
#     echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [3/6] =================="
#     accelerate launch -m lm_eval \
#         --model hf \
#         --model_args pretrained=${MODEL_ID},dtype=auto,trust_remote_code=$TRUST_REMOTE_CODE \
#         --tasks mmlu \
#         --num_fewshot 5 \
#         --batch_size auto \
#         --verbosity DEBUG \
#         --output_path ./${benchmark}.json
    
#     benchmark="truthfulqa"
#     echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [4/6] =================="
#     accelerate launch -m lm_eval \
#         --model hf \
#         --model_args pretrained=${MODEL_ID},dtype=auto,trust_remote_code=$TRUST_REMOTE_CODE \
#         --tasks truthfulqa \
#         --num_fewshot 0 \
#         --batch_size auto \
#         --output_path ./${benchmark}.json
    
#     benchmark="winogrande"
#     echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [5/6] =================="
#     accelerate launch -m lm_eval \
#         --model hf \
#         --model_args pretrained=${MODEL_ID},dtype=auto,trust_remote_code=$TRUST_REMOTE_CODE \
#         --tasks winogrande \
#         --num_fewshot 5 \
#         --batch_size auto \
#         --output_path ./${benchmark}.json
    
#     benchmark="gsm8k"
#     echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [6/6] =================="
#     accelerate launch -m lm_eval \
#         --model hf \
#         --model_args pretrained=${MODEL_ID},dtype=auto,trust_remote_code=$TRUST_REMOTE_CODE \
#         --tasks gsm8k \
#         --num_fewshot 5 \
#         --batch_size auto \
#         --output_path ./${benchmark}.json

#     end=$(date +%s)
#     echo "Elapsed Time: $(($end-$start)) seconds"
    
#     python ../llm-autoeval/main.py . $(($end-$start))

# elif [ "$BENCHMARK" == "lighteval" ]; then
#     git clone https://github.com/huggingface/lighteval.git
#     cd lighteval 
#     pip install '.[accelerate,quantization,adapters]'
#     num_gpus=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)

#     echo "Number of GPUs: $num_gpus"

#     if [[ $num_gpus -eq 0 ]]; then
#         echo "No GPUs detected. Exiting."
#         exit 1

#     elif [[ $num_gpus -gt 1 ]]; then
#         echo "Multi-GPU mode enabled."
#         accelerate launch --multi_gpu --num_processes=${num_gpus} run_evals_accelerate.py \
#         --model_args "pretrained=${MODEL_ID}" \
#         --use_chat_template \
#         --tasks ${LIGHT_EVAL_TASK} \
#         --output_dir="./evals/"

#     elif [[ $num_gpus -eq 1 ]]; then
#         echo "Single-GPU mode enabled."
#         accelerate launch run_evals_accelerate.py \
#         --model_args "pretrained=${MODEL_ID}" \
#         --use_chat_template \
#         --tasks ${LIGHT_EVAL_TASK} \
#         --output_dir="./evals/"
#     else
#         echo "Error: Invalid number of GPUs detected. Exiting."
#         exit 1
#     fi

#     end=$(date +%s)

#     python ../llm-autoeval/main.py ./evals/results $(($end-$start))

# elif [ "$BENCHMARK" == "eq-bench" ]; then
#     git clone https://github.com/EleutherAI/lm-evaluation-harness
#     cd lm-evaluation-harness
#     pip install -e .
#     pip install accelerate

#     benchmark="eq-bench"
#     echo "================== $(echo $benchmark | tr '[:lower:]' '[:upper:]') [1/1] =================="
#     accelerate launch -m lm_eval \
#         --model hf \
#         --model_args pretrained=${MODEL_ID},dtype=auto,trust_remote_code=$TRUST_REMOTE_CODE \
#         --tasks eq_bench \
#         --num_fewshot 0 \
#         --batch_size auto \
#         --output_path ./evals/${benchmark}.json

#     end=$(date +%s)

#     python ../llm-autoeval/main.py ./evals $(($end-$start))

# else
#     echo "Error: Invalid BENCHMARK value. Please set BENCHMARK to 'nous', 'openllm', or 'lighteval'."
# fi

# if [ "$DEBUG" == "False" ]; then
#     runpodctl remove pod $RUNPOD_POD_ID
# fi

# sleep infinity
