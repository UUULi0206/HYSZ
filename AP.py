import pandas as pd

# 读取CSV文件
df = pd.read_csv(r"D:\UUULi\YOLOv11-QUWU\YOLOv11-QUWU\runs\YAN-11-HYSZ\YAN-11-HYSZ\results.csv")

# 提取AP列并重命名
ap_df = df[["epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]]
ap_df.columns = ["epoch", "AP@0.5 (B)", "AP@0.5:0.95 (B)"]

# 保存结果
ap_df.to_csv("class_ap_results.csv", index=False)