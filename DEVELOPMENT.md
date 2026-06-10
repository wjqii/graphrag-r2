# GraphRAG-R1 开发文档

## 1. 项目概述

GraphRAG-R1 是基于 [verl](https://github.com/volcengine/verl) 框架的 Search-R1 架构扩展项目，旨在通过强化学习（GRPO 算法）训练大语言模型学会**何时搜索、如何搜索、如何利用检索结果**来回答多跳推理问题。

核心思路：将检索工具（搜索引擎）作为环境，LLM 作为智能体，通过多轮交互（搜索-阅读-推理-回答）完成 QA 任务，并使用规则奖励函数进行 GRPO 训练。

### 关键特性

- **多轮搜索交互**：LLM 可在推理过程中主动调用搜索引擎，获取外部知识
- **两阶段训练**：Stage 1 学习搜索行为，Stage 2 优化答案质量
- **多种检索后端**：支持 FAISS 密集检索（E5/BGE）、HippoRAG 图检索
- **GRPO 训练算法**：基于 verl 框架的 GRPO（Group Relative Policy Optimization）
- **LoRA 微调**：支持 LoRA 高效微调，降低显存需求

---

## 2. 项目结构

```
graphrag-r1/
├── verl/                                  # verl 框架（魔改版）
│   ├── verl/
│   │   ├── trainer/
│   │   │   ├── main_ppo.py                # 训练入口（Search-R1 魔改版）
│   │   │   ├── ppo/ray_trainer.py         # 训练主循环（Search-R1 魔改版）
│   │   │   └── config/
│   │   │       └── grpo_trainer.yaml      # GRPO 训练配置
│   │   ├── workers/                       # Ray Worker 实现
│   │   │   ├── actor/                     # Actor（策略网络）
│   │   │   ├── critic/                    # Critic（价值网络）
│   │   │   ├── rollout/                   # Rollout（推理生成）
│   │   │   ├── reward_model/              # 奖励模型
│   │   │   └── sharding_manager/          # 分片管理（FSDP/Megatron）
│   │   └── utils/
│   │       ├── reward_score/qa_em.py      # QA 奖励函数（EM/F1/格式奖励等）
│   │       ├── dataset/rl_dataset.py      # RLHF 数据集加载
│   │       └── ...
│   ├── examples/
│   │   └── grpo_trainer/                  # GRPO 训练示例脚本
│   │       ├── run_qwen2_5-7b_lora_stage1.sh   # Stage 1 训练脚本
│   │       ├── run_qwen2_5-7b_lora_stage2.sh   # Stage 2 训练脚本
│   │       └── run_qwen2_5-7b_lora.sh          # 单阶段训练脚本
│   ├── hipporag/                          # HippoRAG 检索器
│   └── data/                              # verl 内部数据
│       ├── merge/                         # 合并训练/测试集
│       ├── hotpotqa/                      # HotpotQA 数据
│       └── wiki_e5/                       # Wikipedia 语料 + FAISS 索引
├── search_r1/                             # Search-R1 核心组件
│   ├── llm_agent/
│   │   ├── generation.py                  # LLMGenerationManager（多轮生成循环）
│   │   └── tensor_helper.py              # Tensor 处理工具
│   └── search/
│       ├── retrieval.py                   # DenseRetriever / BM25Retriever
│       └── retrieval_server.py            # FAISS 检索 FastAPI 服务器
├── eval/                                  # 评估模块
│   ├── qwen_instruct_grpo.py             # 模型推理 + 搜索评估
│   ├── calc_rule.py                       # 规则评估（EM/F1）
│   ├── eval_online.py                     # LLM-as-Judge 评估
│   └── config.json                        # 评估配置
├── retriever_api.py                       # HippoRAG 风格检索 API
├── train_grpo.sh                          # 训练启动脚本
├── retrieval_launch.sh                    # 检索服务启动脚本
└── data/                                  # 完整数据目录
    ├── hotpotqa/                          # HotpotQA 数据集
    └── nq_search/                         # Natural Questions 数据集
```

---

## 3. 核心架构

### 3.1 整体流程

```
                    ┌─────────────────────────────────────────┐
                    │           RayPPOTrainer (Driver)         │
                    │  verl/trainer/ppo/ray_trainer.py        │
                    └───────────┬─────────────────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                  ▼
    ┌─────────────────┐ ┌──────────────┐ ┌────────────────┐
    │  ActorRollout   │ │    Critic    │ │  RefPolicy     │
    │  Worker (Ray)   │ │  Worker(Ray) │ │  Worker (Ray)  │
    │  策略+推理      │ │  价值估计     │ │  参考策略      │
    └────────┬────────┘ └──────────────┘ └────────────────┘
             │
             ▼
    ┌─────────────────────────────────────────────────┐
    │        LLMGenerationManager                      │
    │        search_r1/llm_agent/generation.py         │
    │                                                   │
    │  for step in range(max_turns):                   │
    │    1. LLM 生成 → 解析 <search> 或 <answer>      │
    │    2. 若 <search> → 调用检索 API → 获取文档      │
    │    3. 若 <answer> → 结束                         │
    │    4. 拼接 observation → 继续生成                │
    └──────────────────────┬──────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Retriever API        │
              │   (FastAPI 服务)        │
              │   FAISS / HippoRAG     │
              └────────────────────────┘
```

### 3.2 多轮生成循环

`LLMGenerationManager.run_llm_loop()` 是核心生成循环，流程如下：

1. **初始化**：将 prompt 编码为 input_ids
2. **循环生成**（最多 `max_turns` 轮）：
   - 调用 `actor_rollout_wg.generate_sequences()` 生成响应
   - `_postprocess_responses()` 截断到 `</search>` 或 `</answer>` 标签
   - `execute_predictions()` 解析动作：
     - `<search>query</search>` → 调用 `batch_search()` → 返回 `<information>...</information>`
     - `<answer>ans</answer>` → 标记完成
     - 无效动作 → 返回提示信息
   - `_update_rolling_state()` 拼接新的响应和观察结果
3. **最终生成**：对仍活跃的样本做最后一轮生成
4. **组合输出**：`_compose_final_output()` 拼接完整的 prompt + responses

### 3.3 奖励函数

奖励函数在 `verl/utils/reward_score/qa_em.py` 中实现，支持三种模式：

| 模式 | 奖励组成 | 用途 |
|------|---------|------|
| `stage1` | `format_reward` + `retrieve_w_decay` | 学习搜索行为 |
| `stage2` | `f1_plus` + `format_punishment` | 优化答案质量 |
| `em`（默认） | `compute_score_em` | 精确匹配 |

**Stage 1 奖励详解**：
- `format_reward`：检查 `<answer>` 标签和 `<search>/<information>` 标签一致性（+0.5/+0.5）
- `retrieve_w_decay`：鼓励搜索行为，首次搜索 R0=0.5，后续递减 R0*k^i

**Stage 2 奖励详解**：
- `f1_plus`：`reward = a * f1 * exp(-b * search_count)`，F1 分数带搜索次数惩罚
- `format_punishment`：检查标签格式一致性（+0.5/+0.5）

---

## 4. 检索服务

项目提供两种检索服务：

### 4.1 FAISS 密集检索服务

**文件**：`search_r1/search/retrieval_server.py`

**启动方式**：
```bash
python search_r1/search/retrieval_server.py \
    --index_path /path/to/e5_Flat.index \
    --corpus_path /path/to/wiki-18.jsonl \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --faiss_gpu
```

**API 接口**：
- `POST /retrieve`：批量检索
  - 请求体：`{"queries": ["query1", "query2"], "topk": 3, "return_scores": true}`
  - 响应：`{"result": [[{"document": {...}, "score": 0.95}, ...], ...]}`

**支持的检索模型**：
- E5 系列（`intfloat/e5-base-v2` 等）：自动添加 `query:` / `passage:` 前缀
- BGE 系列（`BAAI/bge-large-en-v1.5` 等）：自动添加指令前缀

### 4.2 HippoRAG 风格检索服务

**文件**：`retriever_api.py`

**启动方式**：
```bash
SAVE_DIR="verl/hipporag/outputs/server" \
DATA_PATH="verl/hipporag/outputs/server/openie_results_ner_qwen7B_4096:latest.json" \
python retriever_api.py
```

**API 接口**：
- `POST /retrieve`：批量检索
- `GET /health`：健康检查

---

## 5. 训练流程

### 5.1 环境准备

```bash
# 设置 Python 路径
export PYTHONPATH="/path/to/graphrag-r1:/path/to/graphrag-r1/verl:$PYTHONPATH"

# 设置 HuggingFace 缓存
export HF_HOME="/path/to/.cache/huggingface/"

# 设置 Ray 临时目录（避免磁盘满）
export RAY_TMPDIR="/path/to/ray_tmp"
```

### 5.2 启动检索服务

```bash
# 方式 1：使用项目脚本
bash retrieval_launch.sh

# 方式 2：手动启动 FAISS 检索服务
python search_r1/search/retrieval_server.py \
    --index_path data/wiki_e5/e5_Flat.index \
    --corpus_path data/hotpotqa/wiki-18.jsonl \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --faiss_gpu

# 方式 3：启动 HippoRAG 检索服务
python retriever_api.py
```

### 5.3 两阶段训练

#### Stage 1：学习搜索行为

```bash
bash verl/examples/grpo_trainer/run_qwen2_5-7b_lora_stage1.sh
```

**关键参数**：
- `+algorithm.reward_mode=stage1`：使用 format_reward + retrieve_w_decay
- `actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct`：基础模型
- `max_turns=3`：最多 3 轮搜索交互
- `retriever.url="http://127.0.0.1:8089/retrieve"`：检索服务地址
- `trainer.total_epochs=1`：训练 1 个 epoch

#### Stage 2：优化答案质量

```bash
# 修改 STAGE1_CHECKPOINT 为 Stage 1 输出的 checkpoint 路径
bash verl/examples/grpo_trainer/run_qwen2_5-7b_lora_stage2.sh
```

**关键参数**：
- `+algorithm.reward_mode=stage2`：使用 f1_plus + format_punishment
- `actor_rollout_ref.model.path=$STAGE1_CHECKPOINT`：从 Stage 1 checkpoint 继续
- `trainer.total_epochs=3`：训练 3 个 epoch

### 5.4 单阶段训练

```bash
bash train_grpo.sh
```

使用 EM 奖励模式，直接训练。

### 5.5 关键训练参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `data.max_prompt_length` | 最大 prompt 长度 | 1024 |
| `data.max_response_length` | 单轮最大响应长度 | 500 |
| `data.max_start_length` | 初始 prompt 最大长度 | 2048 |
| `data.max_obs_length` | 检索结果最大长度 | 500 |
| `max_turns` | 最大搜索交互轮数 | 3 |
| `retriever.url` | 检索服务 URL | `http://127.0.0.1:8089/retrieve` |
| `retriever.topk` | 检索返回文档数 | 3 |
| `algorithm.adv_estimator` | 优势估计方法 | `grpo` |
| `+algorithm.reward_mode` | 奖励模式 | `stage2` |
| `actor_rollout_ref.rollout.n` | GRPO 采样数 | 4 |
| `actor_rollout_ref.actor.kl_loss_coef` | KL 损失系数 | 0.01 |
| `actor_rollout_ref.actor.state_masking` | 是否屏蔽 observation 的梯度 | true |

---

## 6. 评估流程

### 6.1 模型推理评估

使用 `eval/qwen_instruct_grpo.py` 对训练好的模型进行推理：

```python
from eval.qwen_instruct_grpo import run

run(
    input_parquet_path="data/hotpotqa/test.parquet",
    output_jsonl_path="eval/result/qwen_instruct_grpo/hotpotqa.jsonl",
    model_ckpt="verl_checkpoints/xxx/actor/global_step_200/lora_adapter",
    base_model_path="Qwen/Qwen2.5-7B-Instruct",
    search_url="http://127.0.0.1:8089/retrieve",
    max_turns=3,
    max_new_tokens=512,
)
```

该脚本会：
1. 加载 LoRA 模型
2. 对每个问题进行多轮搜索推理
3. 提取 `<answer>` 标签中的预测答案
4. 保存结果为 JSONL 格式

### 6.2 规则评估

使用 `eval/calc_rule.py` 计算 EM/F1 等指标：

```bash
# 修改 eval/config.json 指向结果文件
python eval/calc_rule.py
```

**配置示例**（`eval/config.json`）：
```json
{
    "input_file": "./result/qwen_instruct_grpo/hotpotqa.jsonl",
    "use_em": true,
    "use_cover_em_1": false,
    "use_cover_em_2": false,
    "acc": true,
    "use_f1": true
}
```

**输出指标**：
- EM（Exact Match）：精确匹配率
- F1：Token 级 F1 分数
- Precision / Recall
- 平均检索次数、检索耗时等统计信息

### 6.3 LLM-as-Judge 评估

使用 `eval/eval_online.py` 通过 Qwen 模型判断预测是否正确：

```bash
python eval/eval_online.py
```

---

## 7. 数据格式

### 7.1 训练数据（Parquet 格式）

训练数据为 Parquet 格式，包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `prompt` | list[dict] | 对话 prompt，格式为 `[{"role": "user", "content": "..."}]` |
| `reward_model` | dict | 奖励计算所需信息，如 `{"ground_truth": "answer"}` |
| `data_source` | str | 数据来源标识（如 `hotpotqa`） |

### 7.2 LLM 交互格式

模型在推理过程中使用以下特殊标签：

```
<search>搜索查询</search>          → 触发检索
<information>检索结果</information>  → 检索结果注入
<answer>最终答案</answer>           → 给出答案
```

### 7.3 Prompt 模板

```
Answer the given question. You must conduct reasoning inside
first every time you get new information. After reasoning, if you find you lack
some knowledge, you can call a search engine by <search> query </search> and it
will return the top searched results between <information> and </information>.
You can search as many times as your want. If you find no further external
knowledge needed, you can directly provide the answer inside <answer> and </answer>,
without detailed illustrations.

User: {question}
Assistant: <think>
```

---

## 8. 配置文件详解

### 8.1 GRPO 训练配置

配置文件路径：`verl/verl/trainer/config/grpo_trainer.yaml`

**核心配置块**：

```yaml
data:                    # 数据配置
  train_files / val_files
  max_prompt_length / max_response_length / max_start_length / max_obs_length
  train_batch_size

actor_rollout_ref:       # Actor-Rollout-Ref 联合配置
  model:
    path: Qwen/Qwen2.5-7B-Instruct
    lora_rank / lora_alpha / target_modules
  actor:
    strategy: fsdp
    optim.lr / ppo_mini_batch_size / ppo_micro_batch_size
    use_kl_loss / kl_loss_coef / kl_loss_type
  rollout:
    name: vllm           # 推理引擎
    temperature / n / n_agent
    gpu_memory_utilization
  ref:                   # 参考策略
    fsdp_config.param_offload

algorithm:               # 算法配置
  adv_estimator: grpo
  reward_mode: stage1 / stage2 / em
  no_think_rl: false

retriever:               # 检索配置
  url: "http://127.0.0.1:8089/retrieve"
  topk: 3

trainer:                 # 训练配置
  total_epochs / total_training_steps
  save_freq / test_freq
  n_gpus_per_node / nnodes
  logger: [console, wandb]

max_turns: 3             # 最大搜索轮数
```

---

## 9. 关键代码修改说明

本项目在 verl 框架基础上做了以下关键修改：

### 9.1 `verl/trainer/main_ppo.py`

- 新增 `RewardManager` 类，支持 `stage1`/`stage2`/`em` 三种奖励模式
- 使用 Hydra 加载 `grpo_trainer.yaml` 配置
- 集成 `qa_em` 奖励函数

### 9.2 `verl/trainer/ppo/ray_trainer.py`

- 引入 `LLMGenerationManager` 和 `GenerationConfig`
- 在 `_validate()` 和 `fit()` 中使用多轮生成循环替代单次生成
- 新增 `max_turns`、`retriever` 等配置项
- 新增 `info_mask` 机制：屏蔽 `<information>` 检索结果对策略梯度的贡献

### 9.3 `verl/utils/reward_score/qa_em.py`

- 新增 `compute_score_f1_plus`：F1 分数 + 搜索次数惩罚
- 新增 `compute_score_format_reward`：Stage 1 格式奖励
- 新增 `compute_score_retrieve_w_decay`：Stage 1 搜索行为奖励（递减）
- 新增 `compute_score_format_punishment`：Stage 2 格式惩罚
- `extract_solution`：提取 `<answer>` 标签内容（要求至少出现 2 次，取最后一次）

### 9.4 `search_r1/llm_agent/generation.py`

- `LLMGenerationManager`：核心多轮生成管理器
- `run_llm_loop()`：多轮 for 循环生成
- `execute_predictions()`：解析 `<search>`/`<answer>` 动作并执行
- `batch_search()`：调用检索 API 获取搜索结果
- `_postprocess_responses()`：截断到 `</search>` 或 `</answer>` 标签

---

## 10. 常见问题

### Q1: 训练时检索服务连接失败

确保检索服务已启动且 URL 配置正确：
```bash
# 检查服务是否运行
curl http://127.0.0.1:8089/health

# 检查训练脚本中的 retriever.url 配置
```

### Q2: OOM（显存不足）

- 减小 `ppo_micro_batch_size` 和 `ppo_mini_batch_size`
- 开启 `param_offload`、`grad_offload`、`optimizer_offload`
- 减小 `gpu_memory_utilization`（如 0.2）
- 减小 `max_prompt_length` / `max_response_length`

### Q3: Ray 相关问题

```bash
# 清理 Ray 临时文件
rm -rf /tmp/ray

# 设置 Ray 临时目录
export RAY_TMPDIR="/path/to/ray_tmp"
```

### Q4: 模型不学习搜索行为

- 确认使用 `stage1` 奖励模式训练
- 检查 `max_turns` 是否 >= 2
- 检查检索服务是否正常返回结果

### Q5: 如何切换检索后端

修改训练脚本中的 `retriever.url`：
- FAISS 检索：`http://127.0.0.1:8000/retrieve`（retrieval_server.py 默认端口）
- HippoRAG 检索：`http://127.0.0.1:8089/retrieve`（retriever_api.py 默认端口）

---

## 11. 依赖环境

- Python >= 3.10
- PyTorch >= 2.0
- verl 框架及其依赖
- vLLM（推理引擎）
- Ray（分布式框架）
- FAISS（检索索引）
- sentence-transformers（Embedding 模型）
- FastAPI + uvicorn（检索服务）
- transformers + peft（模型加载 + LoRA）
- Hydra（配置管理）
- wandb（实验追踪）
