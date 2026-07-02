"""Generate a balanced validation set: 50 samples per dataset, 200 total."""
import os
import pandas as pd
from pathlib import Path

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = HERE.parent / "data"

datasets = {
    "hotpotqa": ROOT / "hotpotqa" / "test.parquet",
    "2wikimultihop": ROOT / "2wikimultihop" / "test.parquet",
    "musique": ROOT / "musique" / "test.parquet",
    "popqa": ROOT / "popqa" / "test.parquet",
}

parts = []
n_per_dataset = 50
seed = 42

for name, path in datasets.items():
    if not path.exists():
        print(f"[SKIP] missing: {path}")
        continue

    df = pd.read_parquet(path)
    if "data_source" not in df.columns:
        df["data_source"] = name

    n = min(n_per_dataset, len(df))
    part = df.sample(n=n, random_state=seed)
    parts.append(part)
    print(f"{name}: sampled {n} / {len(df)}")

out = pd.concat(parts, ignore_index=True)
out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

out_path = ROOT / "merged" / "val_200_balanced.parquet"
out_path.parent.mkdir(parents=True, exist_ok=True)
out.to_parquet(out_path, index=False)

print(f"\nsaved: {out_path}")
print(out["data_source"].value_counts())
