from model import generate
from utils import extract_answer

# 1. 直接作答（Direct）
def direct_solve(question):
    prompt = f"""请直接回答这道数学题，只输出最终数字答案。
问题：{question}
答案："""
    resp = generate(prompt, num_return_sequences=1)[0]
    return extract_answer(resp)

# 2. 思维链（CoT）
def cot_solve(question):
    prompt = f"""请一步步推理解决这道数学题，最后给出最终答案。
问题：{question}
推理过程："""
    resp = generate(prompt, num_return_sequences=1)[0]
    return extract_answer(resp)