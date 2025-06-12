#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, BertConfig
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

'''数据预处理为[当前发言前三条+当前发言+当前发言后三条,标签]'''
# 1151.xlsx

def read_data():
    # 读取数据
    # df = pd.read_excel('./1151.xlsx')
    df = pd.read_csv('mooc_2w.csv')
    # df = pd.read_excel('./data/1151.xlsx')
    df['created_at'] = pd.to_datetime(df['created_at'])

    # 根据stu_xh和created_at进行排序
    df.sort_values(by=['stu_xh', 'created_at'], inplace=True)
    final_data = []

    for stu_xh, group in df.groupby('stu_xh'):
        for index, row in group.iterrows():
            # 获取前3条和后3条记录
            prev_3 = group.loc[:index].tail(4)[:-1]  # 不包括当前记录，所以是tail(4)
            next_3 = group.loc[index:].head(4)[1:]  # 不包括当前记录，所以是head(4)和[1:]

            # 初始化combined_texts，先填充7个空字符串
            combined_texts = [''] * 7

            # 根据prev_3和next_3的实际长度填充内容
            start_index_for_prev = 3 - len(prev_3)  # 计算prev_3应该开始的位置
            for i, prev_text in enumerate(prev_3['content'], start=start_index_for_prev):
                combined_texts[i] = prev_text

            combined_texts[3] = row['content']  # 当前发言

            start_index_for_next = 4  # next_3始终从第5个位置开始填充
            for i, next_text in enumerate(next_3['content'], start=start_index_for_next):
                combined_texts[i] = next_text

            # 添加到最终数据列表
            final_data.append((combined_texts, row['label']))
    return final_data


def encode_texts(texts):
    encoded_texts = []
    for text_group in texts:  # text_group是包含7个文本的列表
        encoded_group = {'input_ids': [], 'attention_mask': []}
        for text in text_group:
            if text:
                encoded = tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
                encoded_group['input_ids'].append(encoded['input_ids'].squeeze(0))
                encoded_group['attention_mask'].append(encoded['attention_mask'].squeeze(0))
            else:  # 对于空字符串，使用特殊的编码表示
                encoded_group['input_ids'].append(torch.zeros(512, dtype=torch.long))
                encoded_group['attention_mask'].append(torch.zeros(512, dtype=torch.long))
        # 将编码后的文本段合并为批次维度
        encoded_group['input_ids'] = torch.stack(encoded_group['input_ids'])
        encoded_group['attention_mask'] = torch.stack(encoded_group['attention_mask'])
        encoded_texts.append(encoded_group)
    return encoded_texts


def prepare_dataset(encoded_texts, labels):
    input_ids = torch.stack([et['input_ids'] for et in encoded_texts])
    attention_masks = torch.stack([et['attention_mask'] for et in encoded_texts])
    labels = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(input_ids, attention_masks, labels)


class ScoringModel(nn.Module):
    def __init__(self, base_model_path):
        super(ScoringModel, self).__init__()
        self.base_model = BertModel.from_pretrained(base_model_path)
        self.config = BertConfig.from_pretrained(base_model_path)

        # 多头注意力层
        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.config.hidden_size, num_heads=1,
                                                         batch_first=True)

        # 双向LSTM层
        self.bilstm = nn.LSTM(input_size=self.config.hidden_size, hidden_size=self.config.hidden_size // 2,
                              num_layers=1, batch_first=True, bidirectional=True)

        # 分类器
        self.classifier = nn.Linear(self.config.hidden_size, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        # input_ids和attention_mask形状：[batch_size, 7, seq_length]
        batch_size, num_segments, seq_length = input_ids.size()
        input_ids = input_ids.view(-1, seq_length)
        attention_mask = attention_mask.view(-1, seq_length)

        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_representation = outputs.last_hidden_state[:, 0, :].view(batch_size, num_segments,
                                                                     -1)  # [batch_size, 7, hidden_size]

        # 应用多头注意力
        attention_output, _ = self.multihead_attention(cls_representation, cls_representation, cls_representation)

        # 应用BiLSTM
        # lstm_output, (h_n, c_n) = self.bilstm(attention_output)
        feature_vector = attention_output[:, -1, :]
        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)

        return logits


def test(model, test_dataloader, epoch, device, results):
    model.eval()
    true_labels = []
    predictions = []
    total_loss = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in test_dataloader:
            batch_input_ids, batch_attention_mask, batch_labels = [b.to(device) for b in batch]
            outputs = model(batch_input_ids, batch_attention_mask)
            loss = loss_fn(outputs, batch_labels)
            total_loss += loss.item() * batch_input_ids.size(0)
            _, predicted_labels = torch.max(outputs, dim=1)
            true_labels.extend(batch_labels.cpu().numpy())
            predictions.extend(predicted_labels.cpu().numpy())

    # 计算评价指标
    avg_loss = total_loss / len(test_dataloader.dataset)
    acc = accuracy_score(true_labels, predictions)
    p = precision_score(true_labels, predictions, zero_division=0)
    r = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)

    results['test_accuracy'].append(acc)

    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {acc}, Precision: {p}, Recall: {r}, F1: {f1}")


def train(batch_size, num_epochs):
    final_data = read_data()
    texts = [item[0] for item in final_data]
    labels = [item[1] for item in final_data]
    texts_train, texts_test, scores_train, scores_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    texts_train_encoded = encode_texts(texts_train)
    texts_test_encoded = encode_texts(texts_test)
    train_dataset = prepare_dataset(texts_train_encoded, scores_train)
    test_dataset = prepare_dataset(texts_test_encoded, scores_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    model = ScoringModel(base_pre_model_path)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    model.to(device)

    results = {'train_accuracy': [], 'test_accuracy': []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for batch in tqdm(train_dataloader):
            batch_input_ids, batch_attention_mask, batch_labels = [b.to(device) for b in batch]
            outputs = model(batch_input_ids, batch_attention_mask)
            loss = loss_fn(outputs, batch_labels)
            total_loss += loss.item() * batch_input_ids.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted_labels = torch.max(outputs, dim=1)
            correct_predictions += (predicted_labels == batch_labels).sum().item()
            total_predictions += batch_labels.size(0)

        epoch_loss = total_loss / len(train_dataloader.dataset)
        epoch_acc = correct_predictions / total_predictions
        results['train_accuracy'].append(epoch_acc)

        print(f"训练阶段 Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        print('测试阶段......')
        test(model, test_dataloader, epoch, device, results)
        model.train()

    # 保存结果到Excel
    df_train = pd.DataFrame({
        'Epoch': range(1, num_epochs + 1),
        'Train Accuracy': results['train_accuracy']
    })
    df_test = pd.DataFrame({
        'Test Accuracy': results['test_accuracy']
    })

    # 将训练和测试数据框合并
    df_results = pd.concat([df_train, df_test], axis=1)

    # 将结果保存到一个Excel工作表中
    df_results.to_excel('./results/mooc_without_bli.xlsx', index=False, sheet_name='Results')


if __name__ == "__main__":
    base_pre_model_path = '../bert-base-chinese'
    batch_size = 4
    num_epochs = 20
    tokenizer = BertTokenizer.from_pretrained(base_pre_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设定设备
    train(batch_size, num_epochs)

