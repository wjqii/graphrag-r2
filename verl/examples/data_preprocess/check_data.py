import pandas as pd

df = pd.read_parquet("/agot/verl/data/geo3k/train.parquet")
for idx, row in df.iterrows():
    prompt = row['prompt']
    # if isinstance(prompt, list):
    for msg in prompt:
        # if isinstance(msg.get('content'), list):
        print(f"Index {idx}: content is list -> {msg['content']}")
        break