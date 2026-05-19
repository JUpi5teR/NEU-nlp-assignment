from model import generate
from utils import extract_answer

def tot_solve(question, max_depth=1, n_branch=2):
    try:
        # 极简版思维树，适配小模型！不再复杂递归
        prompt = f"""请你一步一步解决这道数学题，只写关键步骤。
题目：{question}
输出格式：
步骤1：
步骤2：
答案：数字"""

        steps = []
        current = f"开始解题：{question}"
        
        for d in range(max_depth):
            branch_answers = []
            for _ in range(n_branch):
                res = generate(current + "\n下一步：", temperature=0.6, max_new_tokens=150)
                ans = extract_answer(res[0])
                if ans:
                    branch_answers.append(ans)
            if branch_answers:
                best = max(set(branch_answers), key=branch_answers.count)
                current += f"\n步骤{d+1}：得到答案候选 {best}"
                steps.append(best)
        
        final = generate(f"总结所有步骤，给出最终答案：{current}\n答案：", temperature=0.1)
        return extract_answer(final[0]) or "0"
    
    except:
        return "0"