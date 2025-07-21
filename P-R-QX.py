import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
df = pd.read_csv("class_precision_recall.csv")

# 按Recall排序
sorted_df = df.sort_values(by="Recall (B)")

# 提取Precision和Recall
precision = sorted_df["Precision (B)"].values
recall = sorted_df["Recall (B)"].values

# 计算AP（Average Precision）
ap = np.trapz(precision, recall)

# 绘制P-R曲线
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, marker='.', label=f'P-R Curve (AP = {ap:.4f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve for Class B', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("pr_curve_class_b.png", dpi=300, bbox_inches='tight')
plt.show()