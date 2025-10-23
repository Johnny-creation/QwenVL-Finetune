import re
import math
import string
import numpy as np
from difflib import SequenceMatcher
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

rouge = Rouge()
smooth = SmoothingFunction().method1
lemmatizer = WordNetLemmatizer()

def normalize_text(s: str):
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(f"[{re.escape(string.punctuation)}]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def exact_match(a, b):
    return float(normalize_text(a) == normalize_text(b))

def fuzzy_match(a, b):
    a, b = normalize_text(a), normalize_text(b)
    return SequenceMatcher(None, a, b).ratio()

def ANLS(a, b):
    """Approximate Normalized Levenshtein Similarity"""
    a, b = normalize_text(a), normalize_text(b)
    if not a or not b:
        return 0
    dist = SequenceMatcher(None, a, b).ratio()
    return dist

def extract_number(s):
    """提取数字和单位"""
    s = s.replace(",", "")
    match = re.search(r"(-?\d+(\.\d+)?)([a-zA-Z%μ]*)", s)
    if not match:
        return None, None
    val = float(match.group(1))
    unit = match.group(3).lower()
    return val, unit

def unify_unit(val, unit):
    """自动单位归一化，例如 km→m, cm→m"""
    if unit in ["km", "kilometer", "kilometers"]:
        return val * 1000
    elif unit in ["cm"]:
        return val / 100
    elif unit in ["mm"]:
        return val / 1000
    elif unit in ["%"]:
        return val / 100
    else:
        return val

def detect_question_type(q: str):
    """按问题类型路由：yn/class/count/num"""
    q = q.lower()
    if re.search(r"^(is|are|does|do|was|were|has|have|can|will)\b", q):
        return "yn"
    if re.search(r"how many|count|number of|几|多少", q):
        return "count"
    if re.search(r"\d|temperature|distance|area|length|height|km|m|%", q):
        return "num"
    return "class"

def schema_bonus(pred):
    """简单schema检测奖励"""
    if isinstance(pred, str):
        if any(sym in pred for sym in ["{", "}", "[", "]"]):
            try:
                import json
                json.loads(pred)
                return 0.05  # 有效JSON
            except:
                return -0.05  # 无效JSON
    return 0.0

def format_penalty(pred):
    """长度和无效惩罚"""
    if not pred or pred.strip().lower() in ["nan", "none", "null"]:
        return -0.1
    if len(pred) > 128:
        return -0.05
    return 0.0


def route_reward(pred=None, ref=None, question=None, prompts=None, completions=None, **kwargs):
    """按题型路由评分"""
    qtype = detect_question_type(question or "")
    pred_n, ref_n = normalize_text(pred), normalize_text(ref)

    if qtype == "yn":
        return [1.0 if pred_n == ref_n else 0.0]

    elif qtype == "class":
        em = exact_match(pred, ref)
        anls = ANLS(pred, ref)
        return [0.7 * em + 0.3 * anls]

    elif qtype == "count":
        p_num, _ = extract_number(pred)
        r_num, _ = extract_number(ref)
        if p_num is None or r_num is None:
            return 0.0
        eq = 1.0 if int(round(p_num)) == int(round(r_num)) else 0.0
        diff = abs(p_num - r_num)
        return [0.6 * eq + 0.4 * math.exp(-diff / 2.0)]

    elif qtype == "num":
        p_num, p_unit = extract_number(pred)
        r_num, r_unit = extract_number(ref)
        if p_num is None or r_num is None:
            return 0.0
        p_val, r_val = unify_unit(p_num, p_unit), unify_unit(r_num, r_unit)
        rel_err = abs(p_val - r_val) / (abs(r_val) + 1e-8)
        if rel_err <= 0.02:  # 容差 2%
            base = 1.0
        elif rel_err <= 0.05:
            base = 0.8
        else:
            base = math.exp(-rel_err * 5)
        return [base]

    else:
        # 默认 fallback：模糊匹配 + BLEU
        fm = fuzzy_match(pred, ref)
        bl = sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth)
        return [0.6 * fm + 0.4 * bl]


def accuracy_reward(completions, samples=None, **kwargs):
    """主奖励函数"""
    rewards = []
    for i, pred in enumerate(completions):
        if not samples or i >= len(samples):
            rewards.append(0.0)
            continue
        sample = samples[i]
        conv = sample.get("conversations", [])
        question = next((x["value"] for x in conv if x["from"] == "human"), "")
        answer = next((x["value"] for x in conv if x["from"] in ["gpt", "assistant"]), "")
        base = route_reward(pred, answer, question)
        total = base + schema_bonus(pred) + format_penalty(pred)
        rewards.append(max(0.0, min(1.0, total)))  # clamp 0~1
    return rewards
