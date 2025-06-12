# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import csv
import pandas as pd
import re
from prompts import evaluatePrompt
import json

# 定义要读取的csv文件路径
file_path = './filtered_comments.csv'
labels = ['记忆','理解','应用','分析','评价','创造']





# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # 使用pandas的read_csv函数读取CSV文件
    df = pd.read_csv(file_path)
    # print(df)
    df_cleaned = df.dropna()  # 删除任何含有至少一个NaN值的行
    # 或者，仅删除完全由NaN构成的行和重复的行
    df_cleaned = df.dropna(subset=['text'])
    df_cleaned_nosame = df.drop_duplicates(subset=['text'])
    df_id_content = df_cleaned_nosame.loc[:, ['user_id', 'text']]
    l = []
    for i, row in df_id_content.iterrows():
        if len(row.text) < 30:
            continue
        rs = evaluatePrompt(row.text)
        rs = rs['response']
        try:
            start_idx = rs.find('{')
            end_idx = rs.find('}')
            s = rs[start_idx:end_idx+1]
            js = json.loads(s)
            js['id'] = row.user_id
            if js['label'] in labels:
                l.append(js)
        except:
            js = {}
            l.append(js)
        if i % 5 == 0:
            with open('mook_label_list.txt', 'w', encoding='utf-8') as file:
                # 使用json.dumps可以方便地将列表转换为字符串并写入文件，同时保证格式易读
                l_filter = [item for item in l if item != {}]
                print(l_filter)
                file.write(json.dumps(l_filter))



