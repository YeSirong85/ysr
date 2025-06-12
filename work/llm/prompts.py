import sys
from yachalk import chalk

sys.path.append("..")

import json
import ollama
from ollama import Client

client = Client(host='http://localhost:11434')


# def extractConcepts(prompt: str, metadata={}, model="mistral-openorca:latest"):
#     SYS_PROMPT = (
#         "Your task is extract the key concepts (and non personal entities) mentioned in the given context. "
#         "Extract only the most important and atomistic concepts, if  needed break the concepts down to the simpler concepts."
#         "Categorize the concepts in one of the following categories: "
#         "[event, concept, place, object, document, organisation, condition, misc]\n"
#         "Format your output as a list of json with the following format:\n"
#         "[\n"
#         "   {\n"
#         '       "entity": The Concept,\n'
#         '       "importance": The concontextual importance of the concept on a scale of 1 to 5 (5 being the highest),\n'
#         '       "category": The Type of Concept,\n'
#         "   }, \n"
#         "{ }, \n"
#         "]\n"
#     )
#     response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=prompt)
#     try:
#         result = json.loads(response)
#         result = [dict(item, **metadata) for item in result]
#     except:
#         print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
#         result = None
#     return result
#
#
# def graphPrompt(input: str, metadata={}, model="mistral-openorca:latest"):
#     if model == None:
#         model = "mistral-openorca:latest"
#
#     # model_info = client.show(model_name=model)
#     # print( chalk.blue(model_info))
#
#     SYS_PROMPT = (
#         "You are a network graph maker who extracts terms and their relations from a given context. "
#         "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
#         "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n"
#         "Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n"
#         "\tTerms may include object, entity, location, organization, person, \n"
#         "\tcondition, acronym, documents, service, concept, etc.\n"
#         "\tTerms should be as atomistic as possible\n\n"
#         "Thought 2: Think about how these terms can have one on one relation with other terms.\n"
#         "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
#         "\tTerms can be related to many other terms\n\n"
#         "Thought 3: Find out the relation between each such related pair of terms. \n\n"
#         "Format your output as a list of json. Each element of the list contains a pair of terms"
#         "and the relation between them, like the follwing: \n"
#         "[\n"
#         "   {\n"
#         '       "node_1": "A concept from extracted ontology",\n'
#         '       "node_2": "A related concept from extracted ontology",\n'
#         '       "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n'
#         "   }, {...}\n"
#         "]"
#     )
#
#     USER_PROMPT = f"context: ```{input}``` \n\n output: "
#     response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT)
#     try:
#         result = json.loads(response)
#         result = [dict(item, **metadata) for item in result]
#     except:
#         print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
#         result = None
#     return result


def evaluatePrompt(input: str, metadata={}, model="qwen:14b"):
    # response = client.generate(model='qwen:14b', messages=[
    #     {
    #         'role': 'user',
    #         'content': 'Why is the sky blue?',
    #     },
    # ])
    # print("response", response)

    if model == None:
        model = "qwen:14b"
    #     SYS_PROMPT = (
    #         "You are an ai teacher, and your task is to score the students' thinking ability . "
    #         "You are provided with a Chinese context."
    #         "Your scoring criteria are as follows: "
    #         "1. Understand the problem"
    #         "\t1.1. Score 0 for not understanding the question or not answering it"
    #         "\t1.2. 1 point for failure to follow the test requirements or failure to interpret the question"
    #         "\t1.3. 2 points for understanding questions well"
    #         "2. in order to complete the problem set planning strategy arrangement"
    #         "\t2.1. no strategy or plan score 0"
    #         "\t2.2. Score 1 for irrelevant strategies"
    #         "\t2.3. the strategy or plan is very detailed and clear score 2 points"
    #         "The total score is the sum of the scores according to the above criteria"
    #         "For example, if a student scores 2 for understanding problems and 1 for making plans and strategies, the total score is 2+1=3"
    #         "students' score is 3."
    #         "Format your output as only a list of json.Only one field.As follows:"
    #         "[\n"
    #         "   {\n"
    #                 "score:students' score,"
    #             "}\n"
    #          "]"
    #     )
    SYS_PROMPT = (

        ''' ## 角色
           作为AI教师，一个教育技术领域专家。
           ## 任务要求
           你的任务是根据评论文本，结合布鲁姆的认知目标分类法，将评论文本的考核水平分为label，label包括（记忆,理解,应用,分析,评价,创造）其中之一。    
           记忆（Remembering）：要求学生能够记忆并理解具体知识或抽象知识的概念；
           理解（Understanding）：要求学生能以不同的方式说明、推测知识，则表示理解；
           应用（Applying）：在没有说明问题解决模式的情况下，学生正确地把抽象概念运用于适当的情况；
           分析（Analyzing）：要求学生能够分析和理解问题的结构和执行过程；
           评价（Evaluating）：要求学生能够评估问题的有效性和优化的可能性；
           创造（Creating）：要求学生能够创造性地解决问题。
           ## 约束
           不需要输出分类理由。
           考核水平由低到高的顺序是：记忆、理解、应用、分析、评价、创造。
           一条文本有且只有一个分类结果，例如 label为记忆，当一条文本中出现多个认知水平时，以最高阶水平为文本的分类结果。
           ## 结果输出形式如下结构所示
              {
                "label":xxx ,
                "row_text":原始输入的数据
              }
         '''
    )
    USER_PROMPT = f"请给下面学生的文本进行分类: ```{input}``` \n\n output: "
    response = client.generate(model=model, system=SYS_PROMPT, prompt=USER_PROMPT)
    return response
#     response, _ = client.generate(model_name=model, system=SYS_PROMPT)
#     try:
#         result = json.loads(response)
# #         result = [dict(item, **metadata) for item in result]
#
#         print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
#         result = None
#     return result

