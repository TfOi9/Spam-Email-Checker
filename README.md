# 垃圾邮件分类器

## 项目简介

这是一个英语垃圾邮件分类系统，采用机器学习和深度学习技术完成邮件的自动分类。项目不仅实现了多个分类模型，还包含对抗性样本生成和鲁棒性测试。

**核心功能：**
- 英语垃圾邮件自动分类
- 用户界面（Tkinter方案）
- 多代模型实现与对比
- 对抗性样本生成与测试
- 邮件内容预处理
- 中文内容支持（翻译+分类）

## 文件结构

```
Spam-Email-Checker/
├── README.md                                # 项目文档
├── pre.txt                                  # 项目报告
│
├── interface.py                             # Tkinter用户界面（支持中文翻译）
├── english_spam_check.py                    # Gradio Web界面
├── utils.py                                 # 模型加载和预测核心接口
├── load_files.py                            # 数据集读取工具
│
├── adversarial_attack.py                    # 垃圾邮件伪装生成器
├── strip.py                                 # 伪装邮件文本清洗
├── chinese_washer.py                        # 中文文本处理工具
│
├── autocheck.py                             # 自动化测试脚本
│
├── spam_model.joblib                        # 默认分类模型
├── vectorizer.joblib                        # 文本向量化工具
│
├── data/                                    # 完整数据集目录
│   ├── english/
│   │   ├── ham/                             # 正常邮件
│   │   ├── spam/                            # 垃圾邮件
│   │   ├── hard_ham/                        # 难分辨的正常邮件
│   │   ├── reinforced_spam/                 # 强化学习生成的垃圾邮件
│   │   ├── low/medium_reinforced_spam/      # 低/中等强化垃圾邮件
│   │   └── failed_spam/                     # 分类失败的垃圾邮件
│   └── chinese/
│       ├── ham/                             # 中文正常邮件
│       ├── spam/                            # 中文垃圾邮件
│       ├── test/                            # 中文测试集
│       └── stopWords.txt                    # 中文停用词表
│
├── models/
│   ├── model0/                              # 初代模型（基础实现）
│   │   ├── spam_model.joblib
│   │   ├── utils.py
│   │   └── vectorizer.joblib
│   ├── model1/                              # 二代模型（性能改进）
│   │   ├── spam_model.joblib
│   │   ├── utils.py
│   │   └── vectorizer.joblib
│   └── model2/                              # 三代模型（最优版本）
│       ├── spam_model.joblib
│       └── vectorizer.joblib
│
└── adversarial_analysis/                # 伪装样本与测试结果
    ├── adversarial_report_20251102_205251.txt
    ├── adversarial_samples_20251102_205251.csv
    ├── adversarial_training_data_20251102_205251.csv
    └── categorized_samples/
        ├── high_success/                # 高成功率伪装样本
        ├── medium_success/              # 中等成功率伪装样本
        ├── low_success/                 # 低成功率伪装样本
        └── failed/                      # 伪装失败样本
```

## 核心算法

### 1. 朴素贝叶斯 (Naive Bayes)
**原理：** 基于条件概率和独立性假设，计算邮件属于垃圾邮件的概率。

通过训练数据，计算每个词在垃圾邮件和正常邮件中的条件概率，预测新邮件的分类：

$$P(\text{垃圾邮件}|\text{内容}) = \frac{P(\text{内容}|\text{垃圾邮件}) \times P(\text{垃圾邮件})}{P(\text{内容})}$$

- P(垃圾邮件|内容)：已知邮件内容的情况下，该邮件是垃圾邮件的概率
- P(内容|垃圾邮件)：如果是垃圾邮件，出现这些内容的概率
- P(垃圾邮件)：任意邮件是垃圾邮件的先验概率
- P(内容)：出现这样内容的邮件的概率

**"朴素"的含义：** 假设邮件中的每个词相互独立，虽然这在现实中不完全成立，但实践效果良好。

**优点：** 简单高效、可解释性强、适合高维文本数据  
**缺点：** 特征独立性假设过强

### 2. 逻辑回归 (Logistic Regression)
**原理：** 线性判别模型，直接学习输入到输出的映射关系。

将文本转换为TF-IDF特征，通过梯度下降最小化交叉熵损失学习权重，预测邮件属于垃圾邮件的概率：
- 若 $P(y=1|x) \geq 0.5$ → 分类为垃圾邮件
- 否则 → 分类为正常邮件

**优点：** 模型简洁、计算高效、输出概率形式  
**缺点：** 对线性边界敏感、对多重共线性敏感

### 3. 随机森林 (Random Forest)
**原理：** 集成学习方法，多棵决策树通过"投票"做出最终决策。

- 每棵树从不同的数据子集学习（bagging）
- 利用特征随机选择提高模型多样性
- 通过投票机制综合决策

**优点：** 能捕捉复杂模式、抗过拟合、天然处理不平衡数据  
**缺点：** 模型复杂度高、训练时间较长

### 4. 梯度提升 (Gradient Boosting)
**原理：** 序列集成方法，每棵树逐步纠正前面树的错误。

1. 初始化常数预测
2. 迭代过程：
   - 计算当前模型的负梯度（残差）
   - 用一棵决策树拟合这个负梯度
   - 将树的预测乘以学习率加到现有模型上
3. 学习率控制每步的更新幅度

**优点：** 高性能、灵活性强、特征重要性可解释  
**缺点：** 训练时间长、容易过拟合需调参

## 数据集来源

**Spam Assassin Public Corpus**  
https://spamassassin.apache.org/old/publiccorpus/

**训练集：**
- `20021010_easy_ham` - 容易识别的正常邮件
- `20021010_hard_ham` - 难识别的正常邮件  
- `20021010_spam` - 垃圾邮件

**测试集：**
- `20030228_hard_ham` - 难识别的正常邮件
- `20030228_spam` - 垃圾邮件

## 性能对比与测试结果

### 模型演化对比

| 模型版本 | 错误率 | 误判率 | 说明 |
|---------|-------|-------|------|
| Model 0 | - | - | 初代实现 |
| Model 1 | 27.3% | 2.20% | 二代改进 |
| Model 2 | 13.3% | 5.99% | 三代优化（最优） |

**性能指标说明：**
- **错误率**：将垃圾邮件误判为正常邮件的比例（381/1397）
- **误判率**：将正常邮件误判为垃圾邮件的比例（11/501）

### 对抗性样本测试

对Model 1进行100封垃圾邮件的伪装测试，结果如下：

| 伪装级别 | 样本数 | 分类错误率 |
|---------|-------|----------|
| 高成功率 | 31 | 100% (31/31) |
| 中等成功率 | 16 | 62.5% (10/16) |
| 低成功率 | 11 | 27.3% (3/11) |
| 伪装失败 | 42 | 2.4% (1/42) |
| **总体** | **100** | **45%** |

**结论：** 伪装后邮件的分类错误率从27.3%上升到45%，说明分类器对对抗性样本的鲁棒性需要进一步改进。

## 邮件预处理流程

邮件预处理是提高分类准确度的关键步骤：

1. **提取正文** - 去除邮件头、MIME信息、服务器追踪信息
2. **清理HTML标签** - 去除 `<...>` 格式的标签
3. **移除URLs** - 去除超链接
4. **移除邮箱地址** - 去除 `user@domain.com` 格式
5. **文本规范化** - 转换为小写、移除特殊字符
6. **分词与过滤** - 移除停用词（可选）

## 使用方法

### 方案1：Tkinter界面（支持中文）

```bash
python interface.py
```

**特性：**
- 支持输入中文邮件
- 自动翻译中文为英文进行分类
- 实时显示分类结果和垃圾邮件概率

### 方案2：Gradio Web界面

```bash
python english_spam_check.py
```

**特性：**
- 基于Web的现代界面
- 实时反馈处理结果
- 显示文本长度统计

### 自动化测试

```bash
python autocheck.py
```

## 邮件伪装与鲁棒性测试

### 伪装方法

[adversarial_attack.py](adversarial_attack.py) 实现了多种垃圾邮件伪装策略：

1. **词汇替换** - 将高频特征词替换为同义词  
   - `win` → `receive` / `obtain`
   - `free` → `no cost` / `complimentary`
   - `prize` → `award` / `reward`

2. **格式修改** - 删除特征格式
   - 去除连续标点符号
   - 移除过多链接
   - 清理HTML语句

3. **内容融合** - 添加正常邮件特征
   - 插入正常开头：*"Hello, I hope this email finds you well."*
   - 混合正常语句和短语
   - 调整邮件长度

### 对抗性样本分类

生成的样本按成功率分类存储在 [adversarial_analysis/categorized_samples/](adversarial_analysis/categorized_samples/)：

- `high_success/` - 高伪装成功率样本（分类错误率100%）
- `medium_success/` - 中等伪装成功率样本（错误率62.5%）
- `low_success/` - 低伪装成功率样本（错误率27.3%）
- `failed/` - 伪装失败样本（错误率2.4%）

## 中文支持

### 实现方案

项目支持中文邮件处理，有两种方案：

**方案1：翻译后分类（推荐）**
- 使用 `translate` 库将中文翻译为英文
- 使用英文模型进行分类
- 实现在 [interface.py](interface.py)

**方案2：中文模型（未实现）**
- 使用 `jieba` 进行中文分词
- 过滤中文停用词（停用词表在 [data/chinese/stopWords.txt](data/chinese/stopWords.txt)）
- 使用与英文相同的工作流训练中文专用模型

## 依赖库

```
scikit-learn      # 机器学习库
joblib            # 模型序列化
numpy             # 数值计算
pandas            # 数据处理
gradio            # Web界面
translate         # 文本翻译
tkinter           # GUI界面（Python标准库）
```

## 项目成员

**上海交通大学**  
苏成蹊、赵睿城、王文轩  
时间：2025年11月

---

*更新于 2025年12月*
