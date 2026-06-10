import os
import re
import string
import json
import jsonlines
from collections import Counter, defaultdict
import statistics


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation + "".join(["'", "'", "`", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)
    def lower(text):
        return text.lower()
    def replace_underscore(text):
    # 1. 字典脱壳：如果传进来的是字典，提取里面的真实答案文本
        if isinstance(text, dict):
            text = text.get('target', text.get('answer', str(text)))
        
        # 2. 强转字符串，保平安
        text = str(text)
        
        # 3. 标签剥离：精准抠出模型预测的 <answer> 里面的内容，防止 EM 算分被推导过程干扰
        if "<answer>" in text and "</answer>" in text:
            text = text.split("<answer>")[1].split("</answer>")[0].strip()
            
        return text.replace("_", " ")
    return white_space_fix(remove_articles(remove_punc(lower(replace_underscore(s)))))


def bool_mapping(s):
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s


def exact_match_score(prediction, ground_truth):
    return normalize_answer(bool_mapping(prediction)) == normalize_answer(bool_mapping(ground_truth))


def cover_exact_match_score_1(prediction, ground_truth):
    pre_list = normalize_answer(bool_mapping(prediction)).split()
    ground_list = normalize_answer(bool_mapping(ground_truth)).split()
    return all(ground in pre_list for ground in ground_list)


def cover_exact_match_score_2(prediction, ground_truth):
    pre_list = normalize_answer(bool_mapping(prediction)).split()
    ground_list = normalize_answer(bool_mapping(ground_truth)).split()
    for i in range(len(pre_list) - len(ground_list) + 1):
        if pre_list[i:i + len(ground_list)] == ground_list:
            return True
    return " ".join(ground_list) in " ".join(pre_list)


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    normalized_ground_truth = normalize_answer(bool_mapping(ground_truth))

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return 0, 0, 0
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return 0, 0, 0
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores = []
    if metric_fn.__name__ == "f1_score":
        for gt in ground_truths:
            f1, p, r = metric_fn(prediction, gt)
            scores.append((f1, p, r))
        return max(scores, key=lambda x: x[0])
    else:
        for gt in ground_truths:
            score = metric_fn(prediction, gt)
            scores.append(score)
        return max(scores)


def read_jsonl(file_path):
    with jsonlines.open(file_path, "r") as reader:
        return [obj for obj in reader]


def eval(config):
    data = read_jsonl(config["input_file"])
    print(f"Eval {len(data)} examples from {config['input_file']}")

    metrics = {}
    enabled_metrics = {}

    if config.get("use_em", False):
        enabled_metrics["em"] = exact_match_score
        metrics["em"] = 0
    if config.get("use_cover_em_1", False):
        enabled_metrics["cover_em_1"] = cover_exact_match_score_1
        metrics["cover_em_1"] = 0
    if config.get("use_cover_em_2", False):
        enabled_metrics["cover_em_2"] = cover_exact_match_score_2
        metrics["cover_em_2"] = 0
    if config.get("use_f1", False):
        enabled_metrics["f1"] = f1_score
        metrics["f1"] = 0
        metrics["precision"] = 0
        metrics["recall"] = 0

    valid_count = 0
    total_retrieve_num = 0
    retrieve_num_list = []
    total_response_times_sum = 0
    count_1 = 0
    total_generate_times_sum = 0
    total_retrieve_tokens_sum = 0
    total_reasoning_tokens_sum = 0
    total_response_times_count = 0
    total_generate_times_count = 0
    retrieve_num_count = defaultdict(int)

    for d in data:
        pred = d.get("pred_ans", None)
        if not pred or (isinstance(pred, str) and pred.strip() == ""):
            generated = d.get("generated_answer", "")
            if generated:
                match = re.search(r'<answer>(.*?)</answer>', generated, re.DOTALL)
                if match:
                    pred = match.group(1).strip()

        retrieve_num = d.get("retrieve_num", None)
        response_times = d.get("response_times", [])
        generate_times = d.get("generate_times", [])
        retrieve_tokens = d.get("retrieve_tokens", [])
        reasoning_tokens = d.get("reasoning_tokens", [])

        if not pred or (isinstance(pred, str) and pred.strip() == ""):
            continue

        valid_count += 1
        if retrieve_num is not None:
            total_retrieve_num += retrieve_num
            retrieve_num_list.append(retrieve_num)
            count_1 += 1

        if retrieve_num is not None:
            retrieve_num_count[retrieve_num] += 1

        if response_times:
            total_response_times_sum += sum(response_times)
            total_response_times_count += len(response_times)

        if generate_times:
            total_generate_times_sum += sum(generate_times)
            total_generate_times_count += len(generate_times)

        if retrieve_tokens:
            total_retrieve_tokens_sum += sum(retrieve_tokens)

        if reasoning_tokens:
            total_reasoning_tokens_sum += reasoning_tokens[-1]

        gts = d["answer"] if isinstance(d["answer"], list) else [d["answer"]]
        for name, func in enabled_metrics.items():
            if name == "f1":
                f1, p, r = metric_max_over_ground_truths(func, pred, gts)
                metrics["f1"] += f1
                metrics["precision"] += p
                metrics["recall"] += r
            else:
                score = metric_max_over_ground_truths(func, pred, gts)
                metrics[name] += float(score)

    if valid_count == 0:
        print("No valid predictions to evaluate!")
        return metrics

    average_retrieve_num = total_retrieve_num / valid_count if valid_count > 0 else 0
    retrieve_num_variance = statistics.variance(retrieve_num_list) if len(retrieve_num_list) > 1 else 0
    retrieve_num_median = statistics.median(retrieve_num_list) if retrieve_num_list else 0
    overall_average_response_time = total_response_times_sum / total_response_times_count if total_response_times_count > 0 else 0

    print(f"\nValid evaluation samples: {valid_count}")
    print("\nResult:")
    for k, v in metrics.items():
        if k in ["precision", "recall"]:
            continue
        value = round(v / valid_count * 100, 2)
        print(f"{k}: {value}")

    if "precision" in metrics:
        precision_value = round(metrics["precision"] / valid_count * 100, 2)
        print(f"precision: {precision_value}")

    if "recall" in metrics:
        recall_value = round(metrics["recall"] / valid_count * 100, 2)
        print(f"recall: {recall_value}")

    print("\nRetrieve_num statistics:")
    for rn, count in retrieve_num_count.items():
        print(f"  retrieve_num={rn}: {count} samples")

    print(f"\nAverage retrieve count: {average_retrieve_num:.2f}")
    print(f"Retrieve count variance: {retrieve_num_variance:.4f}")
    print(f"Retrieve count median: {retrieve_num_median:.2f}")
    print(f"Average response time per retrieval: {overall_average_response_time:.4f}s")
    print(f"Average response time per sample: {total_response_times_sum / count_1:.4f}s")
    print(f"Average generation time per round: {total_generate_times_sum / total_generate_times_count:.4f}s")
    print(f"Average generation time per sample: {total_generate_times_sum / len(data):.4f}s")
    print(f"Average retrieve tokens per round: {total_retrieve_tokens_sum / total_response_times_count:.2f}")
    print(f"Average retrieve tokens per sample: {total_retrieve_tokens_sum / count_1:.2f}")
    print(f"Average reasoning tokens per round: {total_reasoning_tokens_sum / total_generate_times_count:.2f}")
    print(f"Average reasoning tokens per sample: {total_reasoning_tokens_sum / len(data):.2f}")

    return metrics


if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)
    eval(config)
