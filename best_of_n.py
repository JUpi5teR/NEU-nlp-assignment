from model import generate
from utils import extract_answer, score_reasoning

def best_of_n_solve(question, n=5):
    """
    Best-of-N：生成 N 条推理，选分数最高的答案
    """
    prompt = f"""请一步步推理解决这道数学题，最后给出答案。
问题：{question}
推理过程："""

    responses = generate(prompt, num_return_sequences=n, temperature=0.8)
    
    # 打分排序
    scored = []
    for r in responses:
        ans = extract_answer(r)
        score = score_reasoning(r)
        scored.append((ans, score))
    
    # 选最高分
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0] if scored else None