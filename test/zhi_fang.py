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

# 计算每周的起止时间
weeks = sorted(weekly_stats.index.unique())
week_ranges = []
for week in weeks:
    start_time = start_date + timedelta(days=(week - week_offset - 1) * 7)
    end_time = start_time + timedelta(days=6)
    week_ranges.append(f'{start_time.strftime("%Y-%m-%d")} 到 {end_time.strftime("%Y-%m-%d")}')

# 添加每周的起止时间到weekly_stats中
weekly_stats['week_range'] = week_ranges

# 将结果保存到新的CSV文件
weekly_stats.to_csv('WeeklyStats_with_Range.csv')

# 设置中文字体为宋体
plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用宋体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号'-'显示为方块的问题

# 绘制直方图
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = weekly_stats.index

bar1 = ax.bar(index - bar_width/2, weekly_stats['label_0'], bar_width, label='低阶')
bar2 = ax.bar(index + bar_width/2, weekly_stats['label_1'], bar_width, label='高阶')

# 在每个直方图上面添加数字，并确保数字是整数格式
for bars in [bar1, bar2]:
    for bar in bars:
        height = int(bar.get_height())  # 转换为整数
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# 设置横坐标标签格式
ax.set_xlabel('周数')
ax.set_ylabel('数量')
ax.set_title('每周预测标签分布')
ax.set_xticks(index)
ax.set_xticklabels([f'第{i}周\n({r})' for i, r in zip(index, week_ranges)], rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.show()