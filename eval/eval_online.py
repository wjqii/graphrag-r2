import dashscope
from http import HTTPStatus
import json
from tqdm import tqdm
import os

dashscope.api_key = 'sk-your-api-key-here'


def generate_qwen_response(prompt):
    response = dashscope.Generation.call(
        model='qwen-turbo',
        messages=[{"role": "user", "content": prompt}],
        result_format='message',
    )
    if response.status_code == HTTPStatus.OK:
        return response.output.choices[0].message.content
    else:
        print("Error:", response)
        return ""


def process_one_sample(obj):
    if obj.get("pred_ans", "") == "":
        obj["qwen_output"] = "False"
        return obj

    prompt = '''Given a Question and its Golden Answer, verify whether the Predicted Answer is correct. The prediction is correct if it fully aligns with the meaning and key information of the Golden Answer. Respond with True if the prediction is correct and False otherwise.

    Question: {question}
    Golden Answer: {reference}
    Predicted Answer: {prediction}
    '''

    question = obj["question"]
    reference_ans = obj["answer"]
    prediction = obj["pred_ans"]

    if reference_ans == False:
        reference_ans = "no"
    if reference_ans == True:
        reference_ans = "yes"

    qwen_input = prompt.format(question=question, reference=reference_ans, prediction=prediction)
    response = generate_qwen_response(qwen_input)
    obj["qwen_output"] = response

    return obj


if __name__ == '__main__':
    input_files = [
        "./result/qwen_instruct_grpo/hotpotqa.jsonl",
        "./result/qwen_instruct_grpo/2wiki.jsonl",
        "./result/qwen_instruct_grpo/musique.jsonl",
    ]

    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"File not found, skipping: {input_file}")
            continue
        print(input_file)
        output_file = input_file.replace(".jsonl", "_judge_qwen.jsonl")

        with open(input_file, "r") as fin:
            all_data = [json.loads(line) for line in fin]

        results = []
        for item in tqdm(all_data):
            res = process_one_sample(item)
            results.append(res)

        correct = 0
        total = 0
        for item in results:
            output = item.get("qwen_output", "").lower()
            if "true" in output:
                correct += 1
            total += 1
        accuracy = correct / total if total > 0 else 0

        with open(output_file, "w") as fout:
            for res in results:
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")

            summary = {
                "summary": {
                    "accuracy": round(accuracy, 4),
                    "correct": correct,
                    "total": total,
                }
            }
            fout.write(json.dumps(summary, ensure_ascii=False) + "\n")

        print(f"\nAccuracy: {accuracy:.2%} ({correct}/{total})\n")
