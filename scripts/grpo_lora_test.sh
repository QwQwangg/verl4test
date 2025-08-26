# -*- coding: utf-8 -*-
# export CUDA_VISIBLE_DEVICES=4,5,6,7
NOW=$(date +%Y%m%d)
# export WANDB_DIR=gsm8k-grpo-lora-qwen2.5-32b-${NOW}
# export WANDB_PROJECT=${WANDB_DIR}
# export WANDB_EXP=32b-${NOW}
# MODEL_PATH=/data1/model/Qwen3-30B-A3B-Instruct-2507
export RAY_DEBUG=0
# model
MODEL="Qwen2.5/Qwen2.5-1.5B"
ALL_MODEL="/mnt/d/weiqin/All/models"
MODEL_PATH=$ALL_MODEL/$MODEL
# dataset
DATASET="gsm8k"
ALL_DATA="/mnt/d/weiqin/verl4test/data"
DATA_PATH=$ALL_DATA/$DATASET
# training
GAE="grpo"
LORA_RANK=32
LORA_ALPHA=32
LR=1e-6
EPOCH=15
BATCH_SIZE=8
PROMPT_LENGTH=512
RESPONSE_LENGTH=1024
MINI_BATCH_SIZE=8
MICRO_BATCH_SIZE=1

set -x

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
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.model.use_shm=True  \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.max_model_len=1536 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1536 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "swanlab"]' \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXP} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 $@ 2>&1 | tee ${WANDB_PROJECT}.log