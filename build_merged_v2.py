"""重新构造无泄露的 merged_v2 数据集
- 每个数据集 7:3 划分
- 严格去重（基于问题文本）
- popqa 只做测试集
"""
import pandas as pd
import numpy as np
import hashlib
import os

DATA_DIR = '/agot/graphrag-r1/data'
OUT_DIR = '/agot/graphrag-r1/data/merged_v2'


def get_q_hash(df):
    """用问题文本做唯一标识"""
    def q2str(x):
        if isinstance(x, (list, np.ndarray)) and len(x) > 0:
            if isinstance(x[0], dict):
                return str(x[0].get('content', ''))
            return str(x[0])
        return str(x)
    return df['prompt'].apply(q2str).apply(lambda s: hashlib.md5(s.encode()).hexdigest())


# 1. 加载所有数据集
sources = ['hotpotqa', '2wikimultihop', 'musique', 'popqa']
all_data = {}

for ds in sources:
    parts = []
    for split in ['train', 'test']:
        path = f'{DATA_DIR}/{ds}/{split}.parquet'
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df['_origin_split'] = split
            parts.append(df)
            print(f'{ds}/{split}: {len(df)}')
    if parts:
        merged = pd.concat(parts, ignore_index=True)
        merged['_qhash'] = get_q_hash(merged)
        merged = merged.drop_duplicates(subset='_qhash', keep='first')
        all_data[ds] = merged
        print(f'  → {ds} 去重后: {len(merged)}')

# 2. 按数据集分别 7:3 划分，popqa 全做测试
os.makedirs(OUT_DIR, exist_ok=True)

final_train = []
final_test = []

for ds in sources:
    df = all_data[ds]
    if ds == 'popqa':
        sub = df.drop(columns=['_qhash', '_origin_split']).reset_index(drop=True)
        final_test.append(sub)
        print(f'{ds}: 全做测试 → test {len(sub)}')
    else:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(df)
        n_train = int(n * 0.7)
        tr = df.iloc[:n_train].drop(columns=['_qhash', '_origin_split']).reset_index(drop=True)
        te = df.iloc[n_train:].drop(columns=['_qhash', '_origin_split']).reset_index(drop=True)
        final_train.append(tr)
        final_test.append(te)
        print(f'{ds}: 总 {n} → train {len(tr)} (70%) / test {len(te)} (30%)')

# 3. 合并
final_train_df = pd.concat(final_train, ignore_index=True)
final_test_df = pd.concat(final_test, ignore_index=True)

# 4. 最终校验
final_train_df['_h'] = get_q_hash(final_train_df)
final_test_df['_h'] = get_q_hash(final_test_df)
overlap = set(final_train_df['_h']) & set(final_test_df['_h'])
print(f'\n最终 train ∩ test 重叠: {len(overlap)} 条 (应=0)')
final_train_df = final_train_df.drop(columns=['_h'])
final_test_df = final_test_df.drop(columns=['_h'])

# 5. shuffle
final_train_df = final_train_df.sample(frac=1, random_state=42).reset_index(drop=True)
final_test_df = final_test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 6. 保存
import pyarrow as pa
import pyarrow.parquet as pq

def safe_to_parquet(df, path):
    """处理 reward_model 等混合类型列后再保存"""
    for col in df.columns:
        if df[col].dtype == object:
            # 把可能含 dict 的列转成 JSON 字符串
            try:
                sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                if isinstance(sample, dict):
                    df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else x)
            except Exception:
                pass
    df.to_parquet(path)

import json
safe_to_parquet(final_train_df, f'{OUT_DIR}/train.parquet')
safe_to_parquet(final_test_df, f'{OUT_DIR}/test.parquet')

print(f'\n=== 最终结果 ===')
print(f'训练集: {len(final_train_df)} 条 → {OUT_DIR}/train.parquet')
print(f'测试集: {len(final_test_df)} 条 → {OUT_DIR}/test.parquet')
print(f'\n各数据集分布:')
for ds in sources:
    n_tr = (final_train_df['data_source'] == ds).sum()
    n_te = (final_test_df['data_source'] == ds).sum()
    print(f'  {ds:20s}: train {n_tr:4d}, test {n_te:4d}')
