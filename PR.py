import pandas as pd

# 读取CSV文件
df = pd.read_csv(r"D:\UUULi\YOLOv11-QUWU\YOLOv11-QUWU\runs\YAN-11-HYSZ\YAN-11-HYSZ\results.csv")

# 提取Precision和Recall列并重命名
precision_recall_df = df[["epoch", "metrics/precision(B)", "metrics/recall(B)"]]
precision_recall_df.columns = ["Epoch", "Precision (B)", "Recall (B)"]

# 保存结果
precision_recall_df.to_csv("class_precision_recall.csv", index=False)