# GraphRAG-R1 Extension for verl

基于 verl 框架的 Search-R1 架构扩展，完整迁移自 `/home/zhangziwei6/ycy/graphrag/Search-R1`

## 目录结构

```
/home/zhangziwei6/wujiaqi/graphrag-r1/
├── verl/                              # verl 框架
│   ├── verl/                          # 核心代码
│   │   ├── trainer/                   # 训练器
│   │   │   ├── main_ppo.py            # ← Search-R1 魔改版（入口）
│   │   │   └── ppo/ray_trainer.py    # ← Search-R1 魔改版（训练主循环）
│   │   ├── workers/                   # Worker 实现
│   │   └── utils/reward_score/
│   │       └── qa_em.py               # ← Search-R1 的 QA 精确匹配奖励
│   ├── examples/                      # 示例脚本
│   ├── data/                          # 数据目录
│   │   ├── merge/                     # 训练数据（train.parquet, test.parquet）
│   │   ├── hotpotqa/                  # HotpotQA 数据
│   │   ├── nq_search/                 # Natural Questions 数据
│   │   └── wiki_e5/                   # Wikipedia 语料 + FAISS 索引（134GB）
│   ├── hipporag/                      # HippoRAG 检索器
│   └── checkpoints/                    # → 软链接到原始 checkpoints
├── search_r1/                         # Search-R1 核心组件（完整复制）
│   ├── llm_agent/
│   │   ├── generation.py              # LLMGenerationManager（多轮 for 循环）
│   │   └── tensor_helper.py           # Tensor 处理工具
│   └── search/
│       ├── retrieval_server.py         # FAISS 检索服务器
│       └── retrieval.py               # DenseRetriever
├── data/                             # 完整数据（134GB wiki_e5 + 其他）
├── train_grpo.sh                      # 训练脚本
└── retrieval_launch.sh                 # 检索服务启动脚本
```

## 数据说明

| 目录 | 大小 | 说明 |
|------|------|------|
| `wiki_e5/` | 134GB | Wikipedia 语料 + E5 FAISS 索引 |
| `nq_search/` | 2.4GB | Natural Questions 检索数据 |
| `merge/` | 744KB | 合并的训练/测试集 |
| `hotpotqa/` | 28KB | HotpotQA 数据 |

## 使用方法

1. 启动检索服务：
```bash
cd /home/zhangziwei6/wujiaqi/graphrag-r1
bash retrieval_launch.sh
```

2. 运行训练：
```bash
cd /home/zhangziwei6/wujiaqi/graphrag-r1
bash train_grpo.sh
```

## 依赖

- verl 框架（原版 verl/wujiaqi 的依赖）
- Search-R1 专用组件（已完整复制到此目录）

## 特性

- 多轮搜索交互（LLMGenerationManager）
- FAISS 密集检索（E5 模型）
- GRPO 训练算法
- LoRA 微调支持
