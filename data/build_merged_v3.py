"""
Rebuild train/test parquet for 4 retrieval-augmented QA datasets.

Each dataset is split 80/20 on its OWN rows (not on the merged pool), so that
every dataset is represented in both train and test. This avoids the
"all-popqa-goes-to-test" failure mode of the previous full-pool 7:3 split.

Source columns expected:
    - 'data_source' (label, e.g. 'hotpotqa')
    - 'prompt'      (list of {role, content} messages, used directly by verl)
    - 'reward_model' or 'ground_truth' or 'target' (used for the answer key)
    - 'data_source' for filter

Output:
    /home/zhangziwei6/wujiaqi/graphrag-r1/data/merged_v3/train.parquet
    /home/zhangziwei6/wujiaqi/graphrag-r1/data/merged_v3/test.parquet
    /home/zhangziwei6/wujiaqi/graphrag-r1/data/merged_v3/test_small.parquet
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_ROOT = '/agot/graphrag-r1/data'
OUT_DIR = os.path.join(DATA_ROOT, 'merged_v3')
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = ['hotpotqa', '2wikimultihop', 'musique', 'popqa']
SEED = 42
TEST_RATIO = 0.20
SMALL_TEST_PER_DATASET = 50  # for the cheap eval set

train_dfs = []
test_dfs = []
small_test_dfs = []

for name in DATASETS:
    train_path = os.path.join(DATA_ROOT, name, 'train.parquet')
    test_path = os.path.join(DATA_ROOT, name, 'test.parquet')

    # Prefer the dataset's own train/test if both look non-trivial.
    # Fall back to a fresh 80/20 split on whichever side is larger.
    if os.path.exists(train_path) and os.path.exists(test_path):
        df_train_src = pd.read_parquet(train_path)
        df_test_src = pd.read_parquet(test_path)
        print(f'[load] {name}: train={len(df_train_src)}, test={len(df_test_src)}')
        # Re-split the dataset's own train into (train_part, small_test_part)
        if len(df_train_src) >= 100:
            df_tr, df_small = train_test_split(
                df_train_src,
                test_size=SMALL_TEST_PER_DATASET if SMALL_TEST_PER_DATASET < len(df_train_src) else int(len(df_train_src) * TEST_RATIO),
                random_state=SEED,
            )
        else:
            df_tr = df_train_src
            df_small = df_test_src
        # Final test pool: small_test_part + dataset's original test
        df_te = pd.concat([df_small, df_test_src], ignore_index=True)
    else:
        raise FileNotFoundError(f'missing train/test parquet for {name}')

    df_tr = df_tr.copy()
    df_te = df_te.copy()
    df_tr['data_source'] = name
    df_te['data_source'] = name

    # small_test is the cheap eval set, capped at SMALL_TEST_PER_DATASET
    if len(df_small) > SMALL_TEST_PER_DATASET:
        df_small = df_small.sample(n=SMALL_TEST_PER_DATASET, random_state=SEED).copy()
    df_small['data_source'] = name

    # Normalize reward_model column to a JSON string so all rows share the same schema
    import json
    def _norm_rm(v):
        if isinstance(v, dict):
            target = v.get('ground_truth', '')
            if isinstance(target, dict):
                target = target.get('target', '')
            return json.dumps({'ground_truth': {'target': str(target)}})
        if isinstance(v, str):
            # already a JSON string, normalize its inner shape
            try:
                d = json.loads(v)
            except Exception:
                return json.dumps({'ground_truth': {'target': v}})
            if isinstance(d, dict):
                g = d.get('ground_truth', '')
                if isinstance(g, dict):
                    g = g.get('target', '')
                return json.dumps({'ground_truth': {'target': str(g)}})
            return json.dumps({'ground_truth': {'target': str(d)}})
        return json.dumps({'ground_truth': {'target': ''}})

    for sub in (df_tr, df_te, df_small):
        if 'reward_model' in sub.columns:
            sub['reward_model'] = sub['reward_model'].apply(_norm_rm)
        else:
            sub['reward_model'] = [json.dumps({'ground_truth': {'target': ''}})] * len(sub)

    print(f'[split] {name}: new_train={len(df_tr)}, new_test={len(df_te)}, small_test={len(df_small)}')
    train_dfs.append(df_tr)
    test_dfs.append(df_te)
    small_test_dfs.append(df_small)

train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
test_df = pd.concat(test_dfs, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
small_test_df = pd.concat(small_test_dfs, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

print()
print('=== Final counts ===')
print(f'train:        {len(train_df)}')
print(f'test:         {len(test_df)}')
print(f'test_small:   {len(small_test_df)}')
print()
print('=== train distribution ===')
print(train_df['data_source'].value_counts())
print()
print('=== test_small distribution ===')
print(small_test_df['data_source'].value_counts())

train_df.to_parquet(os.path.join(OUT_DIR, 'train.parquet'), index=False)
test_df.to_parquet(os.path.join(OUT_DIR, 'test.parquet'), index=False)
small_test_df.to_parquet(os.path.join(OUT_DIR, 'test_small.parquet'), index=False)
print()
print(f'[done] wrote to {OUT_DIR}')
