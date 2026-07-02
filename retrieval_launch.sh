#!/bin/bash

cd /agot/graphrag-r1

export CUDA_VISIBLE_DEVICES=4
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export SAVE_DIR="verl/hipporag/outputs/server"
export DATA_PATH="verl/hipporag/outputs/server/openie_results_ner_qwen7B_4096:latest.json"

echo "Starting HippoRAG Retriever..."
echo "Data path: $DATA_PATH"
echo "Embedding model: BAAI/bge-large-en-v1.5"
echo "Port: 8089"

python3 retriever_api.py
