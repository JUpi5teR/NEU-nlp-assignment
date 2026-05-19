from model import generate
from utils import extract_answer, vote_answer

def self_consistency_solve(question, n=5):
    """
    Self-Consistency：生成 N 条 CoT，多数投票得到最终答案
    """
    prompt = f"""请一步步推理解决这道数学题，最后给出答案。
问题：{question}
推理过程："""

    # 采样 N 条推理路径
    responses = generate(prompt, num_return_sequences=n, temperature=0.8)
    
    # 提取每条答案
    answers = [extract_answer(r) for r in responses]
    
    # 投票
    final = vote_answer(answers)
    return final