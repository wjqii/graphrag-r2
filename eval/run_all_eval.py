"""Run calc_rule.py once per dataset jsonl."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["hotpotqa", "2wiki", "musique", "popqa"]
BASE_CONFIG = {
    "use_em": True,
    "use_cover_em_1": False,
    "use_cover_em_2": False,
    "acc": True,
    "use_f1": True,
}

result_dir = "./result/qwen_instruct_grpo"
if len(sys.argv) > 1:
    result_dir = sys.argv[1]

results = {}
for name in DATASETS:
    cfg = dict(BASE_CONFIG)
    cfg["input_file"] = f"{result_dir}/{name}.jsonl"
    cfg_path = os.path.join(HERE, f"_tmp_config_{name}.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"\n========== {name.upper()} ==========")
    result = subprocess.run(
        [sys.executable, "calc_rule.py", cfg_path],
        cwd=HERE,
    )
    results[name] = result.returncode

for name, rc in results.items():
    print(f"{name}: exit code {rc}")

# Cleanup tmp configs
for name in DATASETS:
    p = os.path.join(HERE, f"_tmp_config_{name}.json")
    if os.path.exists(p):
        os.remove(p)
