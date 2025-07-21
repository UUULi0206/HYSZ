import matplotlib.pyplot as plt
import pandas as pd

# 假设数据已加载，这里模拟数据（实际使用需替换为真实数据加载）
data = pd.DataFrame({
    'epoch': range(1, 201),
    'AP@0.5 (B)': [0.0, 0.45773, 0.54245, 0.88843, 0.63358] + [0]*195,
    'AP@0.5:0.95 (B)': [0.0, 0.14793, 0.18262, 0.39742, 0.32938] + [0]*195
})

# 设置字体为楷体
plt.rcParams['font.family'] = 'KaiTi'  # 设置全局字体为楷体
plt.rcParams['figure.dpi'] = 300
plt.figure(figsize=(12, 6))

# 配色调整（深蓝、绿色）
plt.plot(
    data['epoch'],
    data['AP@0.5 (B)'],
    label='AP@0.5 (B)',
    marker='o',
    color='#004488',
    linewidth=2
)
plt.plot(
    data['epoch'],
    data['AP@0.5:0.95 (B)'],
    label='AP@0.5:0.95 (B)',
    marker='s',
    color='#008844',
    linewidth=2
)

plt.title('AP 指标随 epoch 的变化', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.xticks(rotation=45)
plt.ylabel('AP 指标值', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()