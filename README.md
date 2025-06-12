# 🚀 BABF - 认知状态评估模型

<div align="center">
![版本](https://img.shields.io/badge/版本-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![troch](https://img.shields.io/badge/torch-2.0+-orange)
![许可证](https://img.shields.io/badge/许可证-MIT-brightgreen)
![状态](https://img.shields.io/badge/状态-开发中-yellow)

</div>


## 📖 项目简介

BABF有效解决了通用文本分类模型在语义连续性和时序建模方面的局限，为认知状态的自动评估提供了新的技术方案。

### ✨ 核心功能

- 🤖 **多层次特征提取**：局部＋全局
- 📚 **全局特征融合**：融合前后讨论文本间的时序语义特征
- 📤 **自动分类**：实现认知状态的自动分类
- 🔍 **应用前景**：应用于教学平台的认知状态评估

## 🛠️ 技术架构


- **局部特征提取**：基于BERT提取讨论文本的语义特征
- **全局特征融合**：基于注意力机制和BILSTM实现时序语义特征融合
- **教学应用**：将BABF模型应用于教学平台并进行聚类分析

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 相关依赖包

## 📂 项目结构

```
ERAG/
├── data                   # 数据
├── work                   # 核心代码
├── test                   # 应用代码
└── README.md              # 项目说明文档
```

## 📈 未来计划

- [ ] 支持多源数据格式（图片、行为、视频等）
- [ ] 添加自动标注
- [ ] 优化算法，提高评估速度和准确性
- [ ] 增加多元数据分析功能和使用统计报告

<div align="center">
  ⭐ 如果您觉得这个项目有用，请给它一个star！
</div>
