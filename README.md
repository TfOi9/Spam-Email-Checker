# 垃圾邮件分类器
## 简介
这个项目完成了对于英语垃圾邮件的判别，并实现了一个简单的用户界面。我们采用了深度学习的实现方式，
训练了AI模型来完成对垃圾邮件的分类，实现了较好的分类效果。此外，我们还对垃圾邮件进行了伪装并测试。

## 文件结构
```
Spam-Email-Checker/
├── README.md
├── adversarial_analysis                     # 垃圾邮件伪装
│   ├── adversarial_report.txt
│   ├── adversarial_samples.csv
│   ├── adversarial_training_data.csv
│   └── categorized_samples
│       ├── README.txt
│       ├── failed
│       ├── high_success
│       ├── low_success
│       └── medium_success
├── adversarial_attack.py                    # 垃圾邮件伪装源码
├── autocheck.py                             # 自动测试器
├── data                                     # 数据集
├── english_spam_check.py                    # 英语垃圾邮件分类（基于Gradio的界面）
├── interface.py                             # 英语垃圾邮件分类（用户界面）
├── load_files.py                            # 读取数据集源码
├── models                                   # 各代模型
│   ├── model0
│   │   ├── spam_model.joblib
│   │   ├── utils.py
│   │   └── vectorizer.joblib
│   ├── model1
│   │   ├── spam_model.joblib
│   │   ├── utils.py
│   │   └── vectorizer.joblib
│   └── model2
│       ├── spam_model.joblib
│       └── vectorizer.joblib
├── spam_model.joblib
├── strip.py                                 # 伪装邮件清洗源码
├── utils.py                                 # 模型接口源码
└── vectorizer.joblib
```

## 数据集
我们采用了Spam Assassin的垃圾邮件数据集 https://spamassassin.apache.org/old/publiccorpus/

其中，使用了20021010_easy_ham， 20021010_hard_ham 和 20021010_spam作为训练集；
用20030228_hard_ham，20030228_spam 和 20030228_spam进行测试。

## 测试结果

- 错误率（判断垃圾邮件为正常邮件）为27.3%（381/1397）；
- 误判率（判断正常邮件为垃圾邮件）为2.20%（11/501）；
- 测试了垃圾邮件伪装，对上述spam邮件中的前100封进行伪装，有31封为高伪装成功率，16封中等成功率，11封低成功率，42封无法伪装；随后进行测试：高伪装成功率的分类错误率100%（31/31），中伪装成功率的分类错误率62.5%（10/16），低伪装成功率的分类错误率27.3%（3/11），伪装失败的分类错误率2.4%（1/42），总错误率45%（45/100）。
