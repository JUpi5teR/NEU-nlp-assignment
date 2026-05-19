import re
from collections import Counter

def extract_answer(text):
    """从文本中提取最后一个数字作为答案"""
    if not text:
        return None
    nums = re.findall(r'\d+(?:\.\d+)?', text)
    return nums[-1] if nums else None

def vote_answer(answer_list):
    """多数投票：返回出现次数最多的答案"""
    valid = [a for a in answer_list if a]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]

def score_reasoning(text):
    """给推理步骤打分（用于 Best-of-N）"""
    score = 0
    if "=" in text:
        score += 1
    if len(text.split("\n")) >= 2:
        score += 1
    if re.search(r'[+\-*/]', text):
        score += 1
    return score

def calculate_accuracy(preds, golds):
    """计算准确率"""
    correct = 0
    for p, g in zip(preds, golds):
        if p and g and str(p) == str(g):
            correct += 1
    return correct / len(preds) if preds else 0