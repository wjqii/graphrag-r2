set -x

ENGINE=${1:-vllm}

HOME=/agot/graphrag-r1
export PYTHONPATH="/agot/graphrag-r1:/agot/graphrag-r1/verl:$PYTHONPATH"
export HF_HOME="/agot/verl/.cache/huggingface/"
export WANDB_MODE=offline
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_IGNORE_UNHANDLED_ERRORS=1

export RAY_DISABLE_FILE_SYSTEM_MONITOR=1
export RAY_OBJECT_SPILLING_CONFIG='{"type":"mock"}'
export RAY_TMPDIR="/agot/graphrag-r1/ray_tmp"
export RAY_RESOURCES='{"GPU": 1}'

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

# 优先用早期 Stage1，而不是最终 Stage1。
# 不建议用 Stage1 最终 checkpoint，因为日志显示已经 search-only 化。
#
# 方案 A: 从 Stage1 global_step_200 继续（需要先复制 tokenizer 文件到根目录）
#   cp $INIT_CKPT/huggingface/* $INIT_CKPT/
# 方案 B: 如果 checkpoint 格式有问题，直接用 base model（推荐先跑这个验证 reward）
# INIT_CKPT=/agot/graphrag-r1/verl_checkpoints/merged-v3-lora-grpo-qwen2.5-7b-it-stage1/actor/global_step_200
INIT_CKPT=Qwen/Qwen2.5-7B-Instruct

EXP_NAME=merged-v4-lora-grpo-qwen2.5-7b-it-stage2-fast

CUDA_VISIBLE_DEVICES=3 \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.reward_mode=stage2_fast \
    data.train_files=$HOME/data/merged_v3/train.parquet \
    data.val_files=$HOME/data/merged_v3/test_small.parquet \
    data.train_batch_size=2 \
    data.val_batch_size=4 \
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
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    ++actor_rollout_ref.model.lora_rank=64 \
    ++actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules=all-linear \
    +actor_rollout_ref.model.exclude_modules='.*visual.*' \
    actor_rollout_ref.actor.use_kl_loss=True \
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
    trainer.logger='["console","wandb"]' \
    trainer.project_name='Search-R1' \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    +ray_worker_group_gpus_per_node=0 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.total_training_steps=100 \
    trainer.default_local_dir=/agot/graphrag-r1/verl_checkpoints/$EXP_NAME \
    max_turns=5 \
    retriever.url="http://127.0.0.1:8089/retrieve" \
    retriever.topk=5 \
    2>&1 | tee ${EXP_NAME}.log
