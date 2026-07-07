import json, os, random, shutil
import pandas as pd

SOURCE_DIR = "/home/zhangziwei6/ycy/graphrag/GraphRAG-R1/datasets"
OUTPUT_DIR = "/home/zhangziwei6/wujiaqi/graphrag-r1/data/merged_v4"
BACKUP_DIR = "/home/zhangziwei6/wujiaqi/graphrag-r1/data/merged_v3_backup"
SEED = 42
DATASETS = ["hotpotqa", "2wikimultihop", "musique", "popqa"]

PROMPT_TPL = ("Answer the given question. You must conduct reasoning inside <think> and </think> "
"first every time you get new information. After reasoning, if you find you lack some "
"knowledge, you can call a search engine by <search> query </search> and it will return "
"the top searched results between <information> and </information>. You can search as many "
"times as your want. If you find no further external knowledge needed, you can directly "
"provide the answer inside <answer> and </answer>, without detailed illustrations. "
"For example, <answer> Beijing </answer>. Question: {question}\n")

def load_json(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples

def make_row(ds, q, a, idx, split):
    return {
        "data_source": ds,
        "prompt": [{"content": PROMPT_TPL.format(question=q), "role": "user"}],
        "reward_model": json.dumps({"ground_truth": {"target": a}}),
        "question": q,
        "answer": a,
        "ability": "fact-reasoning" if ds == "2wikimultihop" else None,
        "extra_info": {"index": float(idx), "split": split},
        "label": a if ds in ("hotpotqa", "musique") else None,
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Backup merged_v3
    v3 = "/home/zhangziwei6/wujiaqi/graphrag-r1/data/merged_v3"
    if not os.path.exists(BACKUP_DIR) and os.path.exists(v3):
        os.makedirs(BACKUP_DIR)
        for f in os.listdir(v3):
            src = os.path.join(v3, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(BACKUP_DIR, f))
        print(f"[BACKUP] merged_v3 -> {BACKUP_DIR}")

    all_train, all_test = [], []
    print("=== Dataset Split ===")
    for ds in DATASETS:
        path = os.path.join(SOURCE_DIR, ds, "Question.json")
        samples = load_json(path)
        random.seed(SEED + hash(ds) % 10000)
        random.shuffle(samples)
        n_train = int(len(samples) * 0.8)
        tr, te = samples[:n_train], samples[n_train:]
        print(f"  {ds:15s}: total={len(samples)}, train={len(tr)}, test={len(te)}")
        all_train.append(pd.DataFrame([make_row(ds, s["question"], s["answer"], i, "train") for i, s in enumerate(tr)]))
        all_test.append(pd.DataFrame([make_row(ds, s["question"], s["answer"], i, "test") for i, s in enumerate(te)]))

    train_df = pd.concat(all_train, ignore_index=True)
    test_df = pd.concat(all_test, ignore_index=True)

    # test_small: 50 per dataset
    small_parts = [test_df[test_df["data_source"] == ds].head(50) for ds in DATASETS]
    test_small_df = pd.concat(small_parts, ignore_index=True)

    train_df.to_parquet(os.path.join(OUTPUT_DIR, "train.parquet"), index=False)
    test_df.to_parquet(os.path.join(OUTPUT_DIR, "test.parquet"), index=False)
    test_small_df.to_parquet(os.path.join(OUTPUT_DIR, "test_small.parquet"), index=False)

    print(f"\n=== Statistics ===")
    print(f"train.parquet: {len(train_df)} rows")
    for ds in DATASETS:
        c = len(train_df[train_df["data_source"]==ds])
        print(f"  {ds:15s}: {c} ({c/len(train_df)*100:.1f}%)")
    print(f"\ntest.parquet: {len(test_df)} rows")
    for ds in DATASETS:
        c = len(test_df[test_df["data_source"]==ds])
        print(f"  {ds:15s}: {c} ({c/len(test_df)*100:.1f}%)")
    print(f"\ntest_small.parquet: {len(test_small_df)} rows")
    for ds in DATASETS:
        c = len(test_small_df[test_small_df["data_source"]==ds])
        print(f"  {ds:15s}: {c}")

    print(f"\nColumns: {list(train_df.columns)}")
    print(f"prompt[0]: {str(train_df.iloc[0]['prompt'])[:120]}...")
    print(f"reward_model[0]: {train_df.iloc[0]['reward_model']}")
    print(f"\n[DONE] Rebuild successful!")

if __name__ == "__main__":
    main()
