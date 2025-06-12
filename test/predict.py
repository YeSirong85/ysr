#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
import numpy as np
from bab_without_fusion_earlys import ScoringModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
def load_model(model_path, base_model_path, num_classes):
    model = ScoringModel(base_model_path, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

# 编码文本
def encode_texts(texts, tokenizer):
    input_ids = []
    attention_masks = []
    for text in texts:
        if not isinstance(text, str):  # 检查是否为字符串类型
            text = str(text)  # 强制转换为字符串
        encoded = tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
        input_ids.append(encoded['input_ids'].squeeze(0))
        attention_masks.append(encoded['attention_mask'].squeeze(0))
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    return input_ids, attention_masks

# 预测函数
def predict(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            batch_input_ids, batch_attention_mask = [b.to(device) for b in batch]
            outputs = model(batch_input_ids, batch_attention_mask)
            _, predicted_labels = torch.max(outputs, dim=1)
            predictions.extend(predicted_labels.cpu().numpy())
    return predictions

# 主函数
def main():
    # 参数设置
    model_path = 'best_model.pt'  # 训练好的模型路径
    base_model_path = '../bert-base-chinese'  # BERT 预训练模型路径
    num_classes = 2  # 类别数
    batch_size = 4  # 批大小
    new_csv_path = 'BigData_cleaned.csv'  # 新的 CSV 数据文件路径
    output_csv_path = 'BigData_cleaned_predictions.csv'  # 预测结果保存路径

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 tokenizer
    tokenizer = BertTokenizer.from_pretrained(base_model_path)

    # 读取新数据
    df = pd.read_csv(new_csv_path)
    texts = df['text'].fillna('').astype(str).tolist()

    # 编码文本
    input_ids, attention_masks = encode_texts(texts, tokenizer)

    # 准备数据集
    dataset = TensorDataset(input_ids, attention_masks)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # 加载模型
    model = load_model(model_path, base_model_path, num_classes)

    # 进行预测
    predictions = predict(model, dataloader, device)

    # 保存预测结果
    df['predicted_label'] = predictions
    df.to_csv(output_csv_path, index=False)
    print(f"预测结果已保存到 {output_csv_path}")

if __name__ == "__main__":
    main()