import  json

with open('mook_label_list.txt', 'r', encoding='utf-8') as file:
    # 使用json.dumps可以方便地将列表转换为字符串并写入文件，同时保证格式易读
    data = json.load(file)
    print(data)
    print(len(data))