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

# 读取数据，直接使用单句文本
def read_data():
    # df = pd.read_excel('./1151.xlsx')
    df = pd.read_csv('./mooc_2w.csv')
    texts = df['content'].tolist()
    labels = df['label'].tolist()
    return texts, labels

# 编码单句文本
def encode_texts(texts):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded = tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
        input_ids.append(encoded['input_ids'].squeeze(0))
        attention_masks.append(encoded['attention_mask'].squeeze(0))
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    return input_ids, attention_masks

# 准备数据集
def prepare_dataset(input_ids, attention_masks, labels):
    labels = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(input_ids, attention_masks, labels)

# 自定义注意力层
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / np.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        # hidden 形状: [batch_size, 1, hidden_size]
        # encoder_outputs 形状: [batch_size, seq_length, hidden_size]
        batch_size, seq_length, _ = encoder_outputs.size()
        hidden = hidden.repeat(1, seq_length, 1)  # 调整 hidden 形状以匹配 encoder_outputs
        attn_energies = self.score(hidden, encoder_outputs)
        return nn.functional.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # 拼接 hidden 和 encoder_outputs
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)

# 定义模型
class ScoringModel(nn.Module):
    def __init__(self, base_model_path):
        super(ScoringModel, self).__init__()
        self.base_model = BertModel.from_pretrained(base_model_path)
        self.config = BertConfig.from_pretrained(base_model_path)
        self.attention = Attention(self.config.hidden_size)
        self.bilstm = nn.LSTM(input_size=self.config.hidden_size, hidden_size=self.config.hidden_size // 2,
                              num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(self.config.hidden_size, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # 注意力层
        attn_weights = self.attention(sequence_output[:, 0, :].unsqueeze(1), sequence_output)
        context = attn_weights.bmm(sequence_output)

        # BiLSTM 层
        lstm_output, _ = self.bilstm(context)
        feature_vector = lstm_output[:, -1, :]
        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)

        return logits

# 测试函数
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

# 训练函数
def train(batch_size, num_epochs):
    texts, labels = read_data()
    texts_train, texts_test, scores_train, scores_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    input_ids_train, attention_masks_train = encode_texts(texts_train)
    input_ids_test, attention_masks_test = encode_texts(texts_test)
    train_dataset = prepare_dataset(input_ids_train, attention_masks_train, scores_train)
    test_dataset = prepare_dataset(input_ids_test, attention_masks_test, scores_test)
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

    # 保存结果到 Excel
    df_train = pd.DataFrame({
        'Epoch': range(1, num_epochs + 1),
        'Train Accuracy': results['train_accuracy']
    })
    df_test = pd.DataFrame({
        'Test Accuracy': results['test_accuracy']
    })

    # 将训练和测试数据框合并
    df_results = pd.concat([df_train, df_test], axis=1)

    # 将结果保存到一个 Excel 工作表中
    df_results.to_excel('./results/mooc_without_fusion.xlsx', index=False, sheet_name='Results')

if __name__ == "__main__":
    base_pre_model_path = '../bert-base-chinese'
    batch_size = 2
    num_epochs = 10
    tokenizer = BertTokenizer.from_pretrained(base_pre_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设定设备
    train(batch_size, num_epochs)