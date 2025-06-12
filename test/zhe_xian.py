import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# 加载你的CSV文件
df = pd.read_csv('BigData_cleaned_predictions.csv')

# 将'created_at'列转换为datetime类型
df['created_at'] = pd.to_datetime(df['created_at'])

# 找到数据集中的最小日期作为起始日期来计算周数
start_date = df['created_at'].min()

# 周数偏移量，使得最早的一条记录所在的那一周为第3周
week_offset = 2

# 计算每条记录属于第几周
df['week'] = ((df['created_at'] - start_date).dt.days // 7) + 1 + week_offset

# 对每周的predicted_label进行统计
weekly_stats = df.groupby(['week', 'predicted_label']).size().unstack(fill_value=0)

# 重命名列以便清晰展示
weekly_stats.columns = ['label_0', 'label_1']

# 计算每周的高阶认知比例
weekly_stats['total'] = weekly_stats['label_0'] + weekly_stats['label_1']
weekly_stats['high_order_ratio'] = weekly_stats['label_1'] / weekly_stats['total']

# 设置中文字体为宋体
plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用宋体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号'-'显示为方块的问题

# 绘制按周变化的高阶认知比例折线图
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制折线图
line, = ax.plot(weekly_stats.index, weekly_stats['high_order_ratio'], marker='o', label='高阶认知比例')

# 在每个点上标注比例的数值
for idx, txt in enumerate(weekly_stats['high_order_ratio']):
    ax.annotate(f"{txt:.2f}",
                (weekly_stats.index[idx], txt),
                textcoords="offset points", # 文本相对于注释点的位置
                xytext=(0,10), # 点上方10个点
                ha='center') # 水平居中对齐

# 设置横坐标标签格式
ax.set_xlabel('周数')
ax.set_ylabel('比例')
ax.set_title('按周变化的高阶认知比例')
ax.set_xticks(weekly_stats.index)
ax.set_xticklabels([f'第{i}周' for i in weekly_stats.index])
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()