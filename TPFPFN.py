import pandas as pd

# 读取CSV文件
df = pd.read_csv(r'D:\UUULi\YOLOv11-QUWU\YOLOv11-QUWU\runs\YAN-11-HYSZ\YAN-11-HYSZ\results.csv')

# 设置真实目标数N（根据实际情况调整）
N = 1

# 计算TP、FN、FP
df['TP'] = df['metrics/recall(B)'] * N
df['FN'] = N - df['TP']
df['FP'] = df.apply(
    lambda row: (row['TP'] * (1 - row['metrics/precision(B)']) / row['metrics/precision(B)'])
    if row['metrics/precision(B)'] > 0 else float('nan'),
    axis=1
)

# 保存结果
result_df = df[['epoch', 'TP', 'FP', 'FN']]
result_df.to_csv('epoch_tp_fp_fn.csv', index=False)