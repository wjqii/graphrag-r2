#!/bin/bash
# Docker 内执行 v5 训练的启动脚本
# 容器: wujiaqi_verl_v1
# 路径映射: /home/zhangziwei6/wujiaqi/graphrag-r1 -> /agot/graphrag-r1

set -e

echo "=== Docker v5 训练启动 ==="
echo "时间: $(date)"
echo ""

# Docker 内路径
export HOME=/agot/graphrag-r1
export PYTHONPATH="$HOME:$HOME/verl:$PYTHONPATH"

# Conda 环境
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source /home/zhangziwei6/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate r1_evalue1 2>/dev/null || true

# 缓存和临时目录（docker 内）
export HF_HOME="/agot/graphrag-r1/.cache/huggingface"
export TRITON_CACHE_DIR="/tmp/triton_cache_v5"
export RAY_TMPDIR="/tmp/ray_v5"
export TMPDIR=/tmp

# 离线模式 + 代理清理
export HF_HUB_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export VLLM_USAGE_STATS_COLLECTION=0

# 训练配置
INIT_CKPT=Qwen/Qwen2.5-7B-Instruct
EXP_NAME=merged-v5-lora-grpo-qwen2.5-7b-it-stage2-fast-popqa-fix
ENGINE=vllm

cd $HOME

echo "=== 配置确认 ==="
echo "HOME: $HOME"
echo "EXP_NAME: $EXP_NAME"
echo "INIT_CKPT: $INIT_CKPT"
echo "train_files: $HOME/data/merged_v4/train.parquet"
echo "val_files: $HOME/data/merged_v4/test_small.parquet"
echo "total_steps: 100, save_freq: 25, test_freq: -1"
echo ""

# Ray 清理
ray stop --force 2>/dev/null || true
sleep 3

# 启动训练
CUDA_VISIBLE_DEVICES=3 \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.reward_mode=stage2_fast \
    data.train_files=$HOME/data/merged_v4/train.parquet \
    data.val_files=$HOME/data/merged_v4/test_small.parquet \
    data.train_batch_size=2 \
    data.val_batch_size=2 \
    data.max_prompt_length=3072 \
    data.max_response_length=320 \
    data.max_start_length=2048 \
    data.max_obs_length=700 \
    data.shuffle_train_dataloader=True \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    +actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.model.path=$INIT_CKPT \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    ++actor_rollout_ref.model.lora_rank=64 \
    ++actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules=all-linear \
    +actor_rollout_ref.model.exclude_modules='.*visual.*' \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.02 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_agent=4 \
    actor_rollout_ref.rollout.temperature=0.9 \
    +actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +algorithm.use_kl_in_reward=False \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    +trainer.val_before_train=False \
    trainer.logger='["console"]' \
    trainer.project_name='Search-R1' \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.15 \
    +ray_worker_group_gpus_per_node=0 \
    trainer.save_freq=25 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=100 \
    trainer.default_local_dir=$HOME/verl_checkpoints/$EXP_NAME \
    max_turns=5 \
    retriever.url="http://127.0.0.1:8089/retrieve" \
    retriever.topk=5 \
    2>&1 | tee ${EXP_NAME}.log
