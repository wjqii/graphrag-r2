import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import re
import time
import torch
import pandas as pd
import requests
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel, PeftConfig


PROMPT_TEMPLATE = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer> or <answer> Yes </answer>.
User:{question}Assistant: <think>
"""


def extract_question_from_prompt(prompt_list):
    if isinstance(prompt_list, (list, tuple)):
        for msg in prompt_list:
            if isinstance(msg, dict) and msg.get('role') == 'user':
                return msg.get('content', '')
    return ''


def extract_ground_truth(reward_model):
    if isinstance(reward_model, dict):
        return reward_model.get('ground_truth', reward_model.get('target', ''))
    if isinstance(reward_model, str):
        return reward_model
    return str(reward_model)


def search_query(queries, search_url):
    payload = {"queries": queries, "topk": 3}
    max_try = 5
    for try_count in range(max_try):
        try:
            response = requests.post(search_url, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                return result.get("result", [])
            else:
                print(f"Search server returned status {response.status_code}")
        except Exception as e:
            print(f"Search request failed (attempt {try_count+1}): {e}")
    return []


def format_search_result(answer_item):
    doc_content = ''
    if isinstance(answer_item, list):
        doc_content = "\n".join(str(item) for item in answer_item[:3])
    elif isinstance(answer_item, dict):
        docs = answer_item.get('docs', [])
        facts = answer_item.get('facts', [])
        first_three_docs = docs[:3]
        doc_content = "\n".join(first_three_docs)
        facts = facts[:5] if len(facts) > 5 else facts
        facts_strings = [" ".join(fact) if isinstance(fact, list) else str(fact) for fact in facts]
        facts_with_braces = "\n".join(
            [f"({i + 1}) {{ {fact} }}" for i, fact in enumerate(facts_strings)]
        )
        doc_content = facts_with_braces + "\n" + doc_content
    else:
        doc_content = str(answer_item)
    return doc_content[:2048]


def generate_with_retrieval(model, tokenizer, prompt_text, search_url, max_turns=3, max_new_tokens=512):
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.001,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stop_strings=["</search>", "</answer>"],
    )

    current_text = prompt_text
    retrieve_num = 0
    all_response_times = []
    all_generate_times = []

    for turn in range(max_turns + 1):
        inputs = tokenizer(current_text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                tokenizer=tokenizer,
            )
        generate_time = time.time() - start_time
        all_generate_times.append(generate_time)

        prompt_length = input_ids.size(1)
        generated_ids = outputs[:, prompt_length:]
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        answer_match = re.search(r'<answer>(.*?)</answer>', generated_text, re.DOTALL)
        search_match = re.search(r'<search>(.*?)</search>', generated_text, re.DOTALL)

        if answer_match:
            current_text += generated_text
            break

        if search_match and turn < max_turns:
            query = search_match.group(1).strip()
            query = " ".join(query.split())
            if query:
                retrieve_num += 1
                current_text += generated_text

                start_time = time.time()
                search_results = search_query([query], search_url)
                response_time = time.time() - start_time
                all_response_times.append(response_time)

                if search_results and len(search_results) > 0:
                    doc_content = format_search_result(search_results[0])
                    current_text += f"\n\n<information>{doc_content}</information>\n\n"
                else:
                    current_text += "\n\n<information>No relevant information found.</information>\n\n"
            else:
                current_text += generated_text
                break
        else:
            current_text += generated_text
            break

    final_generated = current_text[len(prompt_text):]

    answer_match = re.search(r'<answer>(.*?)</answer>', final_generated, re.DOTALL)
    pred_ans = answer_match.group(1).strip() if answer_match else ""

    return {
        "generated_answer": final_generated.strip(),
        "pred_ans": pred_ans,
        "retrieve_num": retrieve_num,
        "response_times": all_response_times,
        "generate_times": all_generate_times,
    }


def run(
    input_parquet_path,
    output_jsonl_path,
    model_ckpt,
    base_model_path=None,
    search_url="http://127.0.0.1:8089/retrieve",
    max_turns=3,
    max_new_tokens=512,
    num_samples=200,
):
    print("Model checkpoint path:", model_ckpt)

    if base_model_path is None:
        peft_config = PeftConfig.from_pretrained(model_ckpt)
        base_model_path = peft_config.base_model_name_or_path
    print("Base model path:", base_model_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, model_ckpt)
    model.eval()

    df = pd.read_parquet(input_parquet_path)
    print(f"Parquet columns: {df.columns.tolist()}")
    print(f"Total rows: {len(df)}")

    start_index = max(0, len(df) - num_samples)
    df_eval = df.iloc[start_index:]
    print(f"Evaluation data count: {len(df_eval)}")

    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for idx in tqdm(range(len(df_eval)), desc="Generating"):
            row = df_eval.iloc[idx]

            question = extract_question_from_prompt(row.get('prompt', ''))
            if not question:
                question = row.get('question', '')
            if not question:
                print(f"Warning: No question found for row {idx}, skipping")
                continue

            ground_truth_answer = extract_ground_truth(row.get('reward_model', ''))

            prompt = PROMPT_TEMPLATE.format(question=question)

            result = generate_with_retrieval(
                model, tokenizer, prompt, search_url,
                max_turns=max_turns, max_new_tokens=max_new_tokens,
            )

            if result["pred_ans"]:
                print(f"Q{idx}: {result['pred_ans']}")

            output = {
                "question": question,
                "generated_answer": result["generated_answer"],
                "pred_ans": result["pred_ans"],
                "answer": ground_truth_answer,
                "retrieve_num": result["retrieve_num"],
                "response_times": result["response_times"],
                "generate_times": result["generate_times"],
            }
            f.write(json.dumps(output, ensure_ascii=False) + '\n')

    print(f"Finished. Saved to {output_jsonl_path}")


if __name__ == "__main__":
    result_path = './result/qwen_instruct_grpo'
    checkpoint_path = '../verl_checkpoints/hotpotqa-lora-grpo-qwen2.5-7b-it-parallel/actor/global_step_200/lora_adapter'
    base_model_path = 'Qwen/Qwen2.5-7B-Instruct'
    os.makedirs(result_path, exist_ok=True)
    search_url = 'http://127.0.0.1:8089/retrieve'

    run(
        input_parquet_path="../data/hotpotqa/test.parquet",
        output_jsonl_path=result_path + "/hotpotqa.jsonl",
        model_ckpt=checkpoint_path,
        base_model_path=base_model_path,
        search_url=search_url,
    )
    run(
        input_parquet_path="../data/2wikimultihop/test.parquet",
        output_jsonl_path=result_path + "/2wiki.jsonl",
        model_ckpt=checkpoint_path,
        base_model_path=base_model_path,
        search_url=search_url,
    )
    # run(
    #     input_parquet_path="../data/musique/test.parquet",
    #     output_jsonl_path=result_path + "/musique.jsonl",
    #     model_ckpt=checkpoint_path,
    #     base_model_path=base_model_path,
    #     search_url=search_url,
    # )
    # run(
    #     input_parquet_path="../data/popqa/test.parquet",
    #     output_jsonl_path=result_path + "/popqa.jsonl",
    #     model_ckpt=checkpoint_path,
    #     base_model_path=base_model_path,
    #     search_url=search_url,
    # )
