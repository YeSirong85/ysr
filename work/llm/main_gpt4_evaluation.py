from openai import AzureOpenAI
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score,f1_score

def gpt4_label(Q_content):
    client = AzureOpenAI(
        azure_endpoint="https://research-hxl-gpt4.openai.azure.com/",
        api_key="4fcebfd721ba42bea6937c1294e3450a",
        api_version="2023-05-15",
    )

    response = client.chat.completions.create(
        model="gpt4-32k",  # model = "deployment_name".
        messages=[
            {"role": "system", "content": "## 角色"
                        "作为一个教育技术领域专家。"
                        "## 任务要求"
                        "你的任务是根据评论文本，结合布鲁姆的认知目标分类法，将评论文本的考核水平分为label，label包括（记忆,理解,应用,分析,评价,创造）其中之一。"
                        "记忆（Remembering）：要求学生能够记忆并理解具体知识或抽象知识的概念；"
                        "理解（Understanding）：要求学生能以不同的方式说明、推测知识，则表示理解；"
                        "应用（Applying）：在没有说明问题解决模式的情况下，学生正确地把抽象概念运用于适当的情况；"
                        "分析（Analyzing）：要求学生能够分析和理解问题的结构和执行过程；"
                        "评价（Evaluating）：要求学生能够评估问题的有效性和优化的可能性；"
                        "创造（Creating）：要求学生能够创造性地解决问题。"
                        "## 约束"
                        "不需要输出分类理由。"
                        "考核水平由低到高的顺序是：记忆、理解、应用、分析、评价、创造。"
                        "记忆、理解、应用为低阶思维，标签label为0，分析、评价、创造为高阶思维，标签label为1"
                        "一条文本有且只有一个分类结果，例如 label为0，则是低阶思维能力，label为1，则是高阶思维能力，当一条文本中出现多个认知水平时，以最高水平为文本的分类结果。"
                        "## 结果输出为0或者1"
                        "label"},
            {"role": "user",
             "content": Q_content + "上述评论文本根据布鲁姆的认知目标分类法，如何分类。"}
        ],
    )
    res = response.choices[0].message.content
    return res



# 读取原始Excel文件
df = pd.read_csv('data/mooc_2w.csv')
# print(df)
df_id_content = df.loc[:, ['label', 'content']]
# print(df_id_content)
l = []
for i, row in df_id_content.iterrows():
    try:
        rs = int(gpt4_label(row.content))
        if(rs >= 1):
            l.append(1)
        else:
            l.append(0)
    except:
        l.append(0)

    with open('llm_labeled/gpt4_mooc_eva.txt', 'a', encoding='utf-8') as file:
        # 使用json.dumps可以方便地将列表转换为字符串并写入文件，同时保证格式易读
        file.write(str(i+1) + '\n---ac:' + str(accuracy_score(df_id_content['label'].tolist()[0:i+1],l)) + '\n')
        file.write('---list' + str(l) + '\n')
        print(accuracy_score(df_id_content['label'].tolist()[0:i+1],l))
        print(f1_score(df_id_content['label'].tolist()[0:i+1],l))



