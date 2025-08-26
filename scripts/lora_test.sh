
set -x
# debug
# export RAY_DEBUG_POST_MORTEM=1
export RAY_DEBUG=0
# GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPU_MEMORY_UTILIZATION=0.6
N_GPUS_PER_NODE=8
NNODES=1
TENSOR_PARALLEL_SIZE=2
# model
MODEL="Qwen2.5/Qwen2.5-7B"
ALL_MODEL="/mnt/d/weiqin/All/models"
MODEL_PATH=$ALL_MODEL/$MODEL
# dataset
DATASET="gsm8k"
ALL_DATA="/mnt/d/weiqin/verl4test/data"
DATA_PATH=$ALL_DATA/$DATASET
# rollout
ROLLOUT_N=5
# training
GAE="grpo"
LORA_RANK=32
LORA_ALPHA=32
LR=3e-6
EPOCH=15
BATCH_SIZE=8
PROMPT_LENGTH=512
RESPONSE_LENGTH=1024
MINI_BATCH_SIZE=8
MICRO_BATCH_SIZE=1
# logger
DATE=$(date +%Y%m%d)
TIME_TAG=$(date +%H%M%S)
SWANLAB_KEY=4hjWxDRkx9d3CVgKkrr4X
# EXPERIMENT_NAME="lora_test_${MODEL}_${DATASET}"
PROJECT_NAME="lora_test"
LOG_NAME="${GAE}_${MODEL}_${DATASET}_${DATE}_${TIME_TAG}"
OUTPUT_DIR="/mnt/d/weiqin/verl4test/checkpoints/${PROJECT_NAME}/${LOG_NAME}"

# if need to debug, set RAY_DEBUG=1
#    ray_kwargs.RAY_DEBUG=1 \

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$GAE \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/test.parquet \
    data.train_batch_size=$BATCH_SIZE \
    data.max_prompt_length=$PROMPT_LENGTH \
    data.max_response_length=$RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$LOG_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.total_epochs=$EPOCH $@
