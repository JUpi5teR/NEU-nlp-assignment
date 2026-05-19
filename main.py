from datasets import load_dataset
from tqdm import tqdm
from utils import extract_answer, calculate_accuracy

# 导入所有方法
from baseline import direct_solve, cot_solve
from self_consistency import self_consistency_solve
from best_of_n import best_of_n_solve
from tot import tot_solve

# 加载数据集（GSM8K 小学数学）
ds = load_dataset("gsm8k", "main")
test_set = ds["test"].select(range(50))  # 取50条快速测试

# 定义所有测试方法
methods = {
    "Direct": direct_solve,
    "CoT": cot_solve,
    "Self-Consistency": self_consistency_solve,
    "Best-of-N": best_of_n_solve,
    "ToT": tot_solve
}

# 运行实验
results = {}

print("===== 开始测试 5 种方法 =====")
for name, solver in methods.items():
    print(f"\n正在运行：{name}")
    preds = []
    golds = []

    for item in tqdm(test_set):
        q = item["question"]
        a_true = extract_answer(item["answer"])
        a_pred = solver(q)

        preds.append(a_pred)
        golds.append(a_true)

    acc = calculate_accuracy(preds, golds)
    results[name] = acc
    print(f"{name} 准确率：{acc:.2%}")

# 最终结果汇总
print("\n===== 最终实验结果 =====")
for k, v in results.items():
    print(f"{k:<18} {v:.2%}")