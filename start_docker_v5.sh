#!/bin/bash
# 主机端启动 docker v5 训练
# 使用: bash start_docker_v5.sh

set -e

CONTAINER=wujiaqi_verl_v1
SCRIPT_PATH=/agot/graphrag-r1/verl/examples/grpo_trainer/run_docker_v5.sh

echo "=== 1. 启动 Docker 容器 ==="
docker start $CONTAINER
echo "容器状态: $(docker inspect -f '{{.State.Status}}' $CONTAINER)"

echo ""
echo "=== 2. 验证数据集 ==="
docker exec $CONTAINER python3 -c "
import pandas as pd
for f in ['train','test','test_small']:
    df = pd.read_parquet(f'/agot/graphrag-r1/data/merged_v4/{f}.parquet')
    print(f'{f}.parquet: {len(df)} rows')
    if 'data_source' in df.columns:
        print(f'  {dict(df[\"data_source\"].value_counts())}')
"

echo ""
echo "=== 3. 检查 GPU ==="
docker exec $CONTAINER nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv -i 3

echo ""
echo "=== 4. 检查检索服务 ==="
docker exec $CONTAINER curl --noproxy "*" -s -m 5 -o /dev/null -w "Retriever HTTP: %{http_code}, Time: %{time_total}s" http://127.0.0.1:8089/retrieve -X POST -H "Content-Type: application/json" -d '{"queries":["test"],"topk":3}'
echo ""

echo ""
echo "=== 5. 启动 v5 训练 ==="
echo "配置: merged_v4 数据集, 100 steps, save_freq=25, test_freq=-1"
echo "预计时间: ~2-3 小时"
echo ""
docker exec -it $CONTAINER /bin/bash -c "bash $SCRIPT_PATH"
