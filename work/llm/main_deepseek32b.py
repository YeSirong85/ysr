import pandas as pd
import json
from prompts import evaluatePrompt

# 定义要读取的CSV文件路径
file_path = 'mooc.csv'
output_file_path = 'mooc_deepseek_labeled.csv'
labels = ['l', 'h']

# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # 使用pandas的read_csv函数读取CSV文件
    df = pd.read_csv(file_path)

    # 提取所需的列：id, text, create_time
    df_content = df.loc[:, ['id', 'text']]

    # 初始化一个新的 'label' 列，初始值为空字符串
    df_content['label'] = ''

    l = []  # 用于存储处理后的JSON对象

    # 遍历每一行数据
    for i, row in df_content.iterrows():
        # 调用evaluatePrompt函数评估评论内容
        rs = evaluatePrompt(row.text)
        rs = rs['response']
        print("row.text",row.text)
        print("大模型响应结果",rs)  # 打印评估结果

        try:
            # 提取JSON字符串并解析
            start_idx = rs.find('{')
            end_idx = rs.find('}')
            s = rs[start_idx:end_idx + 1]
            js = json.loads(s)

            # 将id、text、created_at和label添加到JSON对象中
            js['id'] = row.id
            js['text'] = row.text
            # js['created_at'] = row.create_time

            # 如果生成的标签在预定义的labels列表中，则更新DataFrame中的label列
            if js['label'] in labels:
                df_content.loc[i, 'label'] = js['label']
                l.append(js)
            else:
                df_content.loc[i, 'label'] = ""  # 如果标签不在预定义列表中，设置为空字符串
        except (json.JSONDecodeError, KeyError) as e:
            # 处理JSON解析错误或键不存在的情况
            print(f"处理行 {i} 时发生错误: {e}")
            js = {
                'id': row.id,
                'text': row.text,
                # 'created_at': row.create_time,
                'label': "l"
            }
            l.append(js)
            df_content.loc[i, 'label'] = "l"  # 设置为空字符串
            # 将更新后的DataFrame保存到新的CSV文件中
            df_content.to_csv(output_file_path, index=False, encoding='utf-8')
            print(f"已将标注后的数据保存到 {output_file_path}")

    # 如果需要保存为JSON文件（可选）
    # with open('labeled_data.json', 'w', encoding='utf-8') as file:
    #     file.write(json.dumps(l, ensure_ascii=False, indent=4))
    # print("已将标注后的数据保存为 JSON 文件")