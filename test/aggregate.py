import pandas as pd

# 加载你的CSV文件
df = pd.read_csv('BigData.csv')

# 将'time'列转换为datetime类型
df['time'] = pd.to_datetime(df['time'])

# 按'id'分组，并对每个分组按照'time'排序后聚合'label'
grouped = df.sort_values('time').groupby('id')['label'].apply(list).reset_index()

# 重命名列以确保'seq'列存在
grouped.columns = ['id', 'seq']

# 对'seq'列中的0和1进行替换
def replace_labels(seq):
    return ["低阶" if item == 0 else "高阶" if item == 1 else item for item in seq]

# 现在应用replace_labels函数到'seq'列
grouped['seq'] = grouped['seq'].apply(replace_labels)

# 将结果保存到新的CSV文件
grouped.to_csv('AggregatedData.csv', index=False)