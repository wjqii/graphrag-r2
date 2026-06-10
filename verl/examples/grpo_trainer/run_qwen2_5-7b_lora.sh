set -x
ENGINE=${1:-vllm}
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS
# bash examples/grpo_trainer/run_qwen2_5_vl-7b_lora.sh
HOME=/agot/verl
export WANDB_MODE=offline
export VLLM_ATTENTION_BACKEND=XFORMERS

EXP_NAME=hotpotqa-lora-grpo-qwen2.5-7b-it


CUDA_VISIBLE_DEVICES=3 \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/hotpotqa/train.parquet \
    data.val_files=$HOME/data/hotpotqa/test.parquet \
    data.train_batch_size=4 \
    data.val_batch_size=4 \
    data.max_prompt_length=1024 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    +actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    +actor_rollout_ref.model.lora_rank=64 \
    +actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    +actor_rollout_ref.model.exclude_modules='.*visual.*' \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    +actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +algorithm.use_kl_in_reward=False \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    +trainer.val_before_train=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='Search-R1' \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    trainer.default_local_dir=/agot/graphrag-r1/verl_checkpoints/$EXP_NAME \
    max_turns=2 \
    retriever.url="http://127.0.0.1:8089/retrieve" \
    retriever.topk=3 \
    2>&1 | tee ${EXP_NAME}.log