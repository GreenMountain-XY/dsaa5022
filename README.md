# DSAA5022 — 以太坊交易行为分析系统

**课程**：DSAA5022 区块链数据分析
**方向**：方案B — 纯数据分析（欺诈检测 + 异常检测 + 聚类）
**团队**：3人
**周期**：2天

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动应用
streamlit run app.py
```

---

## 项目结构

```
dsaa5022/
├── app.py                      # Streamlit 主应用（C整合）
├── data/
│   └── ethereum_fraud.csv      # 数据集（A提供）
├── data_module/                # 数据层（A实现）
│   ├── __init__.py
│   ├── loader.py               # 数据加载
│   └── preprocessor.py         # 特征工程
├── analysis/                   # AI分析层
│   ├── __init__.py
│   ├── fraud_detector.py       # A: 欺诈检测
│   ├── anomaly_detector.py     # B: 异常检测
│   └── cluster_analysis.py     # C: 聚类分析
├── visualization/              # 可视化层
│   ├── __init__.py
│   ├── fraud_charts.py         # A: 欺诈可视化
│   ├── anomaly_charts.py       # B: 异常可视化
│   └── cluster_charts.py       # C: 聚类可视化
├── pages/                      # Streamlit页面
│   ├── __init__.py
│   ├── overview_page.py        # C: 数据概览
│   ├── fraud_page.py           # A: 欺诈检测页
│   ├── anomaly_page.py         # B: 异常检测页
│   └── cluster_page.py         # C: 聚类分析页
├── scripts/
│   └── generate_mock_data.py   # 备用：模拟数据生成
├── requirements.txt
└── README.md
```

---

## 团队分工

| 成员 | 负责模块 | 核心交付 |
|:---|:---|:---|
| **A** | 数据 + 欺诈检测 | `loader.py`, `preprocessor.py`, `fraud_detector.py`, `fraud_charts.py`, `fraud_page.py` |
| **B** | 异常检测 | `anomaly_detector.py`, `anomaly_charts.py`, `anomaly_page.py` |
| **C** | 主框架 + 聚类 + 整合 | `app.py`, `overview_page.py`, `cluster_analysis.py`, `cluster_charts.py`, `cluster_page.py` |

详见：《方案B_团队分工与接口规范_v2.md》

---

## 接口规范

**数据层**（A实现，全员调用）：
```python
from data_module.loader import load_data, get_feature_matrix, get_labels
from data_module.preprocessor import engineer_features, split_data
```

**分析层**（每人独立实现，符合文档中的类签名）

---

## Git 工作流

```bash
# 每次开工前
git pull origin main

# 开发自己的模块
git checkout -b feature/xxx
git add .
git commit -m "feat: xxx"
git push origin feature/xxx

# 发PR让C合并，或C直接在main合并
```

---

*Created: 2026-05-04*
