from model import generate
from utils import extract_answer

def tot_solve(question, max_depth=2, n_branch=2):
    """
    ToT：多步思考 + 多分支探索
    """
    # 初始分支
    branches = ["开始解题："]

    # 逐层扩展
    for d in range(max_depth):
        new_branches = []
        for b in branches:
            prompt = f"""问题：{question}
当前思考：{b}
请继续下一步推理："""
            next_steps = generate(prompt, num_return_sequences=n_branch, temperature=0.7)
            for step in next_steps:
                new_branches.append(b + "\n" + step)
        # 剪枝：保留合理长度
        branches = [b for b in new_branches if len(b) < 600][:n_branch]

    # 用最优分支生成最终答案
    best_branch = branches[0] if branches else ""
    final_prompt = f"""根据以下推理给出这道题的最终数字答案：
问题：{question}
推理：{best_branch}
答案："""

    resp = generate(final_prompt)[0]
    return extract_answer(resp)