from model import generate
from utils import extract_answer

def direct_solve(question):
    prompt = f"请直接给出这道数学题的答案，只输出数字。题目：{question}"
    res = generate(prompt, temperature=0.1)
    return extract_answer(res[0])

def cot_solve(question):
    prompt = f"""请解决这道数学题，一步一步写清楚，最后给出答案。
题目：{question}
输出格式：
推理过程：
答案：数字"""
    res = generate(prompt, temperature=0.4)
    return extract_answer(res[0])