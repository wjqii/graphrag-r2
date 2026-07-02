# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random
import math
from collections import Counter

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the final answer from the last <answer>...</answer> block."""
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if len(matches) == 0:
        return None
    return matches[-1].group(1).strip()


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth (dict with 'target' key or string)
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    if isinstance(ground_truth, str):
        target = ground_truth
    elif isinstance(ground_truth, dict):
        target = ground_truth.get('target', ground_truth)
    else:
        target = str(ground_truth)

    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {target}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if em_check(answer, target):
            return score
        else:
            return format_score


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth (dict with 'target' key or string)
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    if isinstance(ground_truth, str):
        target = ground_truth
    elif isinstance(ground_truth, dict):
        target = ground_truth.get('target', ground_truth)
    else:
        target = str(ground_truth)

    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {target}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, target):
            return score
        else:
            return format_score


def f1_score_tokens(prediction, ground_truth):
    """Compute F1 score between prediction and ground truth."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

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


def compute_score_f1_plus(solution_str, ground_truth, a=2, b=0.1, format_score=0., score=1.):
    """F1+ reward: F1 score with search count penalty.

    reward = a * f1 * exp(-b * search_count)

    This encourages the model to achieve high F1 with fewer searches.
    """
    if isinstance(ground_truth, str):
        target = ground_truth
    elif isinstance(ground_truth, dict):
        target = ground_truth.get('target', ground_truth)
    else:
        target = str(ground_truth)

    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"[f1_plus] Golden answers: {target}")
        print(f"[f1_plus] Extracted answer: {answer}")

    if answer is None:
        return 0

    search_count = solution_str.count("<search>")
    f1, _, _ = f1_score_tokens(answer, target)
    reward = a * f1 * math.exp(-b * search_count)

    if do_print:
        print(f"[f1_plus] search_count={search_count}, f1={f1:.4f}, reward={reward:.4f}")

    return reward


def compute_score_format_reward(solution_str, ground_truth, format_score=0., score=1.):
    """Stage 1 format reward: checks <answer> tag and search/information tag consistency.

    - +0.5 if <answer>...</answer> exists
    - +0.5 if all <search>/<information> tags are properly matched
    - -0.1 if <answer> tags are missing
    - penalty for malformed tags or Chinese in answer
    """
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"[format_reward] solution preview: {solution_str[:200]}")

    # Check if <answer>...</answer> exists
    if "<answer>" not in solution_str or "</answer>" not in solution_str:
        return -0.1

    score_val = 0.5

    # Check tag consistency
    format_punishment = False
    count_search_begin = solution_str.count("<search>")
    count_search_end = solution_str.count("</search>")
    count_info_begin = solution_str.count("<information>")
    count_info_end = solution_str.count("</information>")

    if count_search_begin == count_search_end >= 1 and count_info_begin == count_info_end >= 1:
        pass
    else:
        format_punishment = True

    # Check answer doesn't contain search/document tags
    answer = extract_solution(solution_str=solution_str)
    if answer is not None:
        if "search" in answer or "information" in answer:
            format_punishment = True

    # Check for Chinese characters (excluding document content)
    modified_solution = re.sub(r'<information>.*?</information>', '', solution_str, flags=re.DOTALL)
    have_chinese = any('\u4e00' <= char <= '\u9fff' for char in modified_solution)
    if have_chinese:
        format_punishment = True

    if not format_punishment:
        score_val += 0.5

    if do_print:
        print(f"[format_reward] score={score_val}")

    return score_val


def compute_score_retrieve_w_decay(solution_str, ground_truth, R0=0.5, k=0.5, format_score=0., score=1.):
    """Stage 1 retrieve reward with decay: encourages search behavior with diminishing returns.

    - R0 for the first search
    - R0 * k^i for the i-th subsequent search (decaying)
    - -0.1 if no search at all
    """
    do_print = random.randint(1, 64) == 1

    search_count = solution_str.count("<search>")
    search_end_count = solution_str.count("</search>")

    if search_count < 1 or search_count != search_end_count:
        if do_print:
            print(f"[retrieve_w_decay] No valid search found, score=-0.1")
        return -0.1

    # Calculate reward with decay
    reward = R0
    cnt = search_count - 1
    plus = R0
    while cnt > 0:
        cnt -= 1
        plus = plus * k
        reward += plus

    if do_print:
        print(f"[retrieve_w_decay] search_count={search_count}, R0={R0}, k={k}, reward={reward:.4f}")

    return reward


def compute_score_format_punishment(solution_str, ground_truth, format_score=0., score=1.):
    """Format punishment reward: checks tag consistency and answer format.

    - +0.5 if <answer>...</answer> exists
    - +0.5 if all tags are properly matched (<search>/<information> pairs)
    - -0.1 if <answer> tags are missing
    - penalty for malformed tags or Chinese in answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"[format_punishment] Extracted answer: {answer}")

    # Check if <answer>...</answer> exists
    if "<answer>" not in solution_str or "</answer>" not in solution_str:
        return -0.1

    score_val = 0.5

    # Check tag consistency
    format_punishment = False
    count_search_begin = solution_str.count("<search>")
    count_search_end = solution_str.count("</search>")
    count_info_begin = solution_str.count("<information>")
    count_info_end = solution_str.count("</information>")

    if count_search_begin == count_search_end >= 1 and count_info_begin == count_info_end >= 1:
        pass
    else:
        format_punishment = True

    # Check answer doesn't contain search/document tags
    if answer is not None:
        if "search" in answer or "information" in answer:
            format_punishment = True

    if not format_punishment:
        score_val += 0.5

    if do_print:
        print(f"[format_punishment] score={score_val}")

    return score_val


def _to_target_list(ground_truth):
    """Normalize ground truth into a list of answer strings."""
    if isinstance(ground_truth, dict):
        target = ground_truth.get("target", ground_truth.get("answer", ground_truth))
    else:
        target = ground_truth
    if isinstance(target, (list, tuple)):
        return [str(x) for x in target]
    return [str(target)]


def _max_f1_em(answer, targets):
    """Compute best F1/EM/precision/recall across multiple targets."""
    best_f1 = 0.0
    best_em = 0.0
    best_precision = 0.0
    best_recall = 0.0
    for target in targets:
        f1, p, r = f1_score_tokens(answer, target)
        em = 1.0 if em_check(answer, target) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_precision = p
            best_recall = r
        best_em = max(best_em, em)
    return best_f1, best_em, best_precision, best_recall


def _evidence_hit(solution_str, targets):
    """Weak evidence reward: whether retrieved information contains any gold answer."""
    infos = re.findall(r'<information>(.*?)</information>', solution_str, flags=re.DOTALL)
    if not infos:
        return 0.0
    info_text = normalize_answer(" ".join(infos))
    for target in targets:
        t = normalize_answer(str(target))
        if t and t in info_text:
            return 1.0
    return 0.0


def compute_score_stage2_fast(solution_str, ground_truth, format_score=0., score=1.):
    """Fast-fix Stage2 reward: F1-driven, no <answer> => negative, format small weight."""
    targets = _to_target_list(ground_truth)
    answer = extract_solution(solution_str)
    search_count = solution_str.count("<search>")
    invalid_count = solution_str.count("My previous action is invalid")

    do_print = random.randint(1, 64) == 1

    # Hard punishment: search-only trajectories must not get positive reward.
    if answer is None or answer.strip() == "":
        reward = -0.5 - 0.05 * search_count - 0.2 * invalid_count
        if do_print:
            print(f"--------------------------------")
            print(f"[stage2_fast] No <answer> found. search={search_count}, invalid={invalid_count}, reward={reward:.4f}")
        return max(reward, -1.0)

    f1, em, _, _ = _max_f1_em(answer, targets)
    evidence = _evidence_hit(solution_str, targets)

    # Main reward
    reward = 2.0 * f1
    reward += 0.5 * em
    reward += 0.2 * evidence
    reward += 0.2  # final answer exists

    # Mild penalties
    reward -= 0.05 * max(0, search_count - 3)
    reward -= 0.2 * invalid_count

    if do_print:
        print(f"--------------------------------")
        print(f"[stage2_fast] targets={targets}")
        print(f"[stage2_fast] answer={answer}")
        print(f"[stage2_fast] f1={f1:.4f}, em={em}, evidence={evidence}, search={search_count}, invalid={invalid_count}")
        print(f"[stage2_fast] reward={reward:.4f}")

    return max(reward, -1.0)
