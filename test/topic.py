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

# 合并第9周和第10周为话题1，第11周和第12周为话题2
topic1_weeks = [9, 10]
topic2_weeks = [11, 12]

topic1_stats = weekly_stats.loc[topic1_weeks].sum()
topic2_stats = weekly_stats.loc[topic2_weeks].sum()

# 创建一个新的DataFrame来存储话题统计数据
topics_stats = pd.DataFrame({
    'label_0': [topic1_stats['label_0'], topic2_stats['label_0']],
    'label_1': [topic1_stats['label_1'], topic2_stats['label_1']],
}, index=['云数据库', 'MapReduce'])

# 计算每个话题的总数量和高阶比例
topics_stats['total'] = topics_stats['label_0'] + topics_stats['label_1']
topics_stats['high_order_ratio'] = topics_stats['label_1'] / topics_stats['total']

# 设置中文字体为宋体
plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用宋体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号'-'显示为方块的问题

# 绘制直方图：每个话题的文本数量分布
fig1, ax1 = plt.subplots(figsize=(8, 6))
bar_width = 0.35
index = topics_stats.index

bar1 = ax1.bar(index, topics_stats['label_0'], bar_width, label='低阶')
bar2 = ax1.bar(index, topics_stats['label_1'], bar_width, bottom=topics_stats['label_0'], label='高阶')

# 在每个直方图上面添加数字，并确保数字是整数格式
for i, (label_0, label_1) in enumerate(zip(topics_stats['label_0'], topics_stats['label_1'])):
    total = int(label_0 + label_1)
    ax1.annotate(f'{total}',
                 xy=(index[i], total),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom')

ax1.set_xlabel('话题')
ax1.set_ylabel('数量')
ax1.set_title('话题文本数量分布')
ax1.legend()
plt.tight_layout()
plt.show()

# 绘制折线图：每个话题的高阶占比
fig2, ax2 = plt.subplots(figsize=(8, 6))

ax2.plot(topics_stats.index, topics_stats['high_order_ratio'], marker='o', label='高阶认知比例')

# 在每个点上标注比例的数值
for idx, txt in enumerate(topics_stats['high_order_ratio']):
    ax2.annotate(f"{txt:.2f}",
                (topics_stats.index[idx], txt),
                textcoords="offset points", # 文本相对于注释点的位置
                xytext=(0,10), # 点上方10个点
                ha='center') # 水平居中对齐

ax2.set_xlabel('讨论主题')
ax2.set_ylabel('比例')
ax2.set_title('不同讨论主题的高阶认知占比')
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.show()