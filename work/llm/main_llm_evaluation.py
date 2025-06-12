import csv
import pandas as pd
import re
from prompts import evaluatePrompt1
import json
from sklearn.metrics import accuracy_score,f1_score

import os
from openai import OpenAI




# 定义要读取的csv文件路径
file_path = 'data/mooc_2w.csv'
labels = ['记忆','理解','应用','分析','评价','创造']





# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    client = OpenAI(
        # This is the default and can be omitted
        api_key="9c96213315a34274809e569b1fd45c8e",
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ],
        model="gpt-3.5-turbo",
        # stream=True
    )
    # for chunk in stream:
    #     if chunk.choices[0].delta.content is not None:
    #         print(chunk.choices[0].delta.content, end="")

    # 使用pandas的read_csv函数读取CSV文件
    # df = pd.read_excel(file_path)
    # df = pd.read_csv(file_path)
    # print(df)
    # df_cleaned = df.dropna()  # 删除任何含有至少一个NaN值的行
    # 或者，仅删除完全由NaN构成的行和重复的行
    # df_cleaned = df.dropna(subset=['text'])
    # df_cleaned_nosame = df.drop_duplicates(subset=['text'])
    # df_id_content = df_cleaned_nosame.loc[:, ['id', 'text']]
    # df_id_content = df_cleaned.loc[:, ['id', 'text']]
    # df_id_content = df.loc[:, ['label', 'content']]
    # l = []
    # for i, row in df_id_content.iterrows():
    #     rs = evaluatePrompt1(row.content,'gemma2:27b')
    #     rs = rs['response']
    #     print(rs)
    #     try:
    #         start_idx = rs.find('{')
    #         end_idx = rs.find('}')
    #         s = rs[start_idx:end_idx+1]
    #         js = json.loads(s)
    #         label = int(js['label'])
    #         if(label >= 1):
    #             l.append(1)
    #         else:
    #             l.append(0)
    #     except:
    #         l.append(0)
    #
    #     with open('gemma2_27b_mooc_eva.txt', 'a', encoding='utf-8') as file:
    #         # 使用json.dumps可以方便地将列表转换为字符串并写入文件，同时保证格式易读
    #         file.write(str(i+1) + '\n---ac:' + str(accuracy_score(df_id_content['label'].tolist()[0:i+1],l)) + '\n')
    #         file.write('---list' + str(l) + '\n')
    #         print(accuracy_score(df_id_content['label'].tolist()[0:i+1],l))
    #         print(f1_score
    #               (df_id_content['label'].tolist()[0:i+1],l))
