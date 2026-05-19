# AI-Financial-Risk-Prediction-System

AI 金融风险预测系统｜Python / Machine Learning / Streamlit

## 项目简介

这是一个面向 GitHub 作品集、实习求职和面试讲解的完整项目，围绕 **Credit Risk Prediction** 与 **Financial Risk Management** 场景展开。项目基于模拟信贷数据完成从数据生成、EDA 分析、模型训练、多模型评估、Explainable AI 到 Streamlit Dashboard 部署的全流程。

## 项目背景

金融机构在信贷业务中需要提前识别高风险客户，以辅助授信审核、风险管理和客户分层。真实银行数据通常不可公开，因此本项目使用符合金融常识的模拟数据来展示完整建模思路与工程实现能力。

## 项目亮点

- 使用 `numpy` 和 `pandas` 生成符合业务逻辑的模拟金融风控数据
- 完成数据分析、相关性分析和多张可视化图表输出
- 对比 `Logistic Regression`、`Random Forest`、`Gradient Boosting`，并在可用时自动加入 `XGBoost`
- 使用 `ROC-AUC`、`F1-score`、`Precision`、`Recall`、`Confusion Matrix` 等指标评估模型
- 输出特征重要性并支持可选 `SHAP` 分析，体现 **Explainable AI**
- 基于 `Streamlit` 构建交互式风险预测系统
- 项目结构规范，适合直接上传 GitHub 展示

## 技术栈

- Python
- Pandas / NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Streamlit Dashboard
- Joblib
- Optional: XGBoost / SHAP

## 项目结构

```text
AI-Financial-Risk-Prediction-System/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── financial_risk_data.csv
├── models/
│   └── best_model.pkl
├── notebooks/
│   └── financial_risk_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── utils.py
└── outputs/
    ├── figures/
    └── reports/
```

## 数据字段说明

| 字段 | 含义 |
|---|---|
| age | 年龄 |
| income | 年收入 |
| loan_amount | 贷款金额 |
| credit_score | 信用评分 |
| employment_years | 工作年限 |
| debt_to_income_ratio | 债务收入比 |
| overdue_times | 历史逾期次数 |
| loan_term | 贷款期限（月） |
| has_house | 是否有房，0/1 |
| has_car | 是否有车，0/1 |
| default | 是否违约，目标变量，0/1 |

## 建模方法

本项目属于二分类任务：

- `default = 1`：高风险 / 违约客户
- `default = 0`：低风险 / 非违约客户

建模流程包括：

1. 读取或生成模拟数据
2. 进行 EDA 和图表输出
3. 划分训练集与测试集
4. 使用 `Pipeline` 组织预处理和模型
5. 训练多个模型并对比效果
6. 按 `ROC-AUC` 选择最佳模型
7. 保存模型与评估结果
8. 输出特征重要性和可解释性结果

## 模型评估指标

项目输出以下指标：

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Classification Report

## 可视化展示

训练流程会自动生成以下图表并保存到 `outputs/figures/`：

- 违约与非违约样本分布图
- 信用评分分布图
- 收入分布图
- 贷款金额分布图
- 债务收入比分布图
- 信用评分与违约风险关系图
- 债务收入比与违约风险关系图
- 收入与违约风险关系图
- 相关性热力图
- ROC 曲线图
- 混淆矩阵图
- 特征重要性图
- 可选 SHAP Summary 图

## Streamlit 系统功能

系统包含四个模块：

1. 项目介绍
2. 数据分析
3. 风险预测
4. 模型解释

在风险预测页面，用户可以输入客户画像，系统会返回：

- 预测结果
- 违约概率
- 风险等级
- 风险提示语
- 简要风险原因解释

## 安装依赖

```bash
pip install -r requirements.txt
```

## 项目运行方式

1. 生成模拟数据

```bash
python src/data_generator.py
```

2. 训练模型并生成图表、报告

```bash
python src/train_model.py
```

3. 启动 Streamlit 可视化系统

```bash
streamlit run app.py
```

完整推荐顺序：

```bash
pip install -r requirements.txt
python src/data_generator.py
python src/train_model.py
streamlit run app.py
```

## GitHub 上传方法

```bash
git init
git add .
git commit -m "feat: build AI financial risk prediction system"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

## 简历项目写法

**AI 金融风险预测系统｜Python / Machine Learning / Streamlit**

基于模拟信贷数据构建金融风险预测模型，完成数据生成、数据分析、特征工程、模型训练、模型评估与可解释性分析。使用 Logistic Regression、Random Forest、Gradient Boosting 等模型预测客户违约风险，并通过 ROC-AUC、F1-score、混淆矩阵等指标评估模型效果。进一步开发 Streamlit 可视化系统，实现客户信息输入、违约概率预测、风险等级判断和特征重要性展示，可用于金融风控、信贷审核和客户风险分层场景。

## 面试可讲重点

- 为什么风控任务常用 `ROC-AUC` 而不仅是 Accuracy
- 为什么 `overdue_times`、`credit_score`、`debt_to_income_ratio` 是关键风险信号
- 为什么要做多模型对比
- 为什么需要 Explainable AI 来辅助业务解释
- 如何把模型结果转化为审核建议和风险分层

## 后续优化方向

- 引入更真实的特征工程，例如负债结构、还款历史窗口统计、行为评分卡变量
- 增加交叉验证、超参数搜索和阈值优化
- 接入真实数据库或 API
- 加入模型监控、漂移检测和再训练流程
- 引入 LightGBM、CatBoost 等模型做更细致对比
- 扩展为 FastAPI + Streamlit 的前后端分离项目

## 免责声明

本项目使用模拟数据，仅用于学习、展示和作品集，不构成真实金融机构授信或投资决策建议。
