#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

'''数据预处理为[当前发言前三条+当前发言+当前发言后三条,标签]'''

def read_data():
    # 读取数据
    # df = pd.read_excel('./non_empty_labels1.xlsx')
    df = pd.read_csv('mooc_2w.csv')
    df['created_at'] = pd.to_datetime(df['created_at'])

    # 根据stu_xh和created_at进行排序
    df.sort_values(by=['stu_xh', 'created_at'], inplace=True)
    final_data = []

    for stu_xh, group in df.groupby('stu_xh'):
        for index, row in group.iterrows():
            # 获取前3条和后3条记录
            prev_3 = group.loc[:index].tail(4)[:-1]  # 不包括当前记录，所以是tail(4)
            next_3 = group.loc[index:].head(4)[1:]   # 不包括当前记录，所以是head(4)和[1:]

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

def encode_texts(texts, tokenizer):
    encoded_texts = []
    for text_group in texts:  # text_group是包含7个文本的列表
        encoded_group = {'input_ids': []}
        for text in text_group:
            if text:
                encoded = tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
                encoded_group['input_ids'].append(encoded['input_ids'].squeeze(0))
            else:  # 对于空字符串，使用特殊的编码表示
                encoded_group['input_ids'].append(torch.zeros(512, dtype=torch.long))
        # 将编码后的文本段合并为批次维度
        encoded_group['input_ids'] = torch.stack(encoded_group['input_ids'])
        encoded_texts.append(encoded_group)
    return encoded_texts

def prepare_dataset(encoded_texts, labels):
    input_ids = torch.stack([et['input_ids'] for et in encoded_texts])
    labels = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(input_ids, labels)

class SimpleEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(SimpleEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, hidden_size)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        pooled = torch.mean(embedded, dim=1)  # 简单的池化操作
        output = self.fc(pooled)
        return output

class ScoringModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(ScoringModel, self).__init__()
        # 简单编码器
        self.encoder = SimpleEncoder(vocab_size, embedding_dim, hidden_size)

        # 多头注意力层
        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1, batch_first=True)

        # 双向LSTM层
        self.bilstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=1, batch_first=True, bidirectional=True)

        # 分类器
        self.classifier = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids):
        # input_ids形状：[batch_size, 7, seq_length]
        batch_size, num_segments, seq_length = input_ids.size()
        input_ids = input_ids.view(-1, seq_length)

        outputs = self.encoder(input_ids)
        cls_representation = outputs.view(batch_size, num_segments, -1)  # [batch_size, 7, hidden_size]

        # 应用多头注意力
        attention_output, _ = self.multihead_attention(cls_representation, cls_representation, cls_representation)

        # 应用BiLSTM
        lstm_output, (h_n, c_n) = self.bilstm(attention_output)
        feature_vector = lstm_output[:, -1, :]
        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)

        return logits

def test(model, test_dataloader, epoch):
    # 验证过程
    model.eval()
    true_labels = []
    predictions = []
    with torch.no_grad():
        for batch in test_dataloader:
            batch_input_ids, batch_labels = [b.to(device) for b in batch]
            outputs = model(batch_input_ids)
            _, predicted_labels = torch.max(outputs, dim=1)

            true_labels.extend(batch_labels.cpu().numpy())
            predictions.extend(predicted_labels.cpu().numpy())

    # 计算评价指标
    acc = accuracy_score(true_labels, predictions)
    p = precision_score(true_labels, predictions, zero_division=0)
    r = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)

    print(f"Epoch {epoch + 1}, Accuracy: {acc}, Precision: {p}, Recall: {r}, F1: {f1}")

def train(batch_size, num_epochs, vocab_size, embedding_dim, hidden_size):
    final_data = read_data()

    # 分离文本和标签
    texts = [item[0] for item in final_data]  # item[0]是包含7个文本的列表
    labels = [item[1] for item in final_data]

    # 划分训练集测试集
    texts_train, texts_test, scores_train, scores_test = \
        train_test_split(texts, labels, test_size=0.2, random_state=42)

    # 编码训练集和测试集的文本数据
    texts_train_encoded = encode_texts(texts_train, tokenizer)
    texts_test_encoded = encode_texts(texts_test, tokenizer)

    # 准备TensorDataset
    train_dataset = prepare_dataset(texts_train_encoded, scores_train)
    test_dataset = prepare_dataset(texts_test_encoded, scores_test)

    # 创建DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # 初始化模型和优化器
    model = ScoringModel(vocab_size, embedding_dim, hidden_size)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    model.to(device)

    results = {'train_accuracy': [], 'test_accuracy': []}

    # 训练模型
    model.train()
    for epoch in range(num_epochs):
        correct_predictions = 0
        total_predictions = 0
        for batch in tqdm(train_dataloader):
            batch_input_ids, batch_labels = [b.to(device) for b in batch]

            # 模型预测
            outputs = model(batch_input_ids)

            # 计算损失
            loss = loss_fn(outputs, batch_labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            '''计算准确率'''
            _, predicted_labels = torch.max(outputs, dim=1)  # 获取最大概率的预测标签
            correct_predictions += (predicted_labels == batch_labels).sum().item()  # 累计正确预测的数量
            total_predictions += batch_labels.size(0)  # 累计预测的总数量

        # 计算并打印这个epoch的平均准确率
        epoch_acc = correct_predictions / total_predictions
        print(f"训练阶段 Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Accuracy: {epoch_acc:.4f}")
        print('测试阶段......')
        test(model, test_dataloader, epoch)
        model.train()
        # 检查训练和测试准确率列表长度是否一致
        assert len(results['train_accuracy']) == len(results['test_accuracy']), "训练和测试准确率列表长度不一致"

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
        df_results.to_excel('./results/mooc_without_bert.xlsx', index=False, sheet_name='Results')

if __name__ == "__main__":
    base_pre_model_path = '../bert-base-chinese'
    batch_size = 4
    num_epochs = 20
    tokenizer = BertTokenizer.from_pretrained(base_pre_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设定设备

    # 假设词汇表大小，实际使用时需要根据tokenizer获取
    vocab_size = tokenizer.vocab_size
    embedding_dim = 300
    hidden_size = 256

    train(batch_size, num_epochs, vocab_size, embedding_dim, hidden_size)