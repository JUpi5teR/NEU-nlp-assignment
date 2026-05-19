import os
from tqdm import tqdm
from utils import extract_answer, calculate_accuracy

# 中文离线题库（适配小模型）
test_set = [
    {"question": "一个班有5个小组，每组20人，总共有多少人？", "answer": "100"},
    {"question": "小明有15个苹果，送朋友6个，还剩多少个？", "answer": "9"},
    {"question": "一本书25元，买3本需要多少钱？", "answer": "75"},
    {"question": "48块饼干分给6个小朋友，每人几块？", "answer": "8"},
    {"question": "汽车每小时60公里，3小时跑多远？", "answer": "180"},
    {"question": "一周7天，4周多少天？", "answer": "28"},
    {"question": "一盒12支铅笔，4盒共几支？", "answer": "48"},
    {"question": "30颗糖吃掉10颗，剩几颗？", "answer": "20"},
    {"question": "一个玩具18元，5个几元？", "answer": "90"},
    {"question": "24只鸟分3个笼子，每笼几只？", "answer": "8"}
]

# 导入方法
from baseline import direct_solve, cot_solve
from self_consistency import self_consistency_solve
from best_of_n import best_of_n_solve
from tot import tot_solve

methods = {
    "Direct": direct_solve,
    "CoT": cot_solve,
    "Self-Consistency": self_consistency_solve,
    "Best-of-N": best_of_n_solve,
    "ToT": tot_solve
}

# 确保输出文件夹存在
os.makedirs("output", exist_ok=True)

print("===== 开始实验 =====")
results = {}

for name, solver in methods.items():
    print(f"\n正在运行：{name}")
    preds = []
    golds = []
    log = []

    for idx, item in enumerate(tqdm(test_set), 1):
        q = item["question"]
        true_a = item["answer"]
        pred_a = solver(q) or "0"

        preds.append(str(pred_a))
        golds.append(str(true_a))

        log.append(f"第{idx}题")
        log.append(f"题目：{q}")
        log.append(f"标准答案：{true_a}")
        log.append(f"模型答案：{pred_a}")
        log.append(f"是否正确：{str(pred_a)==str(true_a)}")
        log.append("-"*40)

    # 保存到独立 txt
    with open(f"output/{name}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log))

    acc = calculate_accuracy(preds, golds)
    results[name] = acc
    print(f"{name} 准确率：{acc:.2%}")

# 输出最终结果
print("\n===== 最终实验结果 =====")
for k, v in results.items():
    print(f"{k:<18} {v:.2%}")

with open("output/实验结果汇总.txt", "w", encoding="utf-8") as f:
    f.write("实验结果汇总\n")
    for k, v in results.items():
        f.write(f"{k}: {v:.2%}\n")