import streamlit as st
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_module.loader import load_data, get_feature_matrix, get_labels
from data_module.preprocessor import engineer_features, split_data
from analysis.fraud_detector import FraudDetector
from analysis.anomaly_detector import AnomalyDetector
from analysis.cluster_analysis import ClusterAnalyzer
from ui.overview_page import show_overview
from ui.fraud_page import show_fraud_detection
from ui.anomaly_page import show_anomaly_detection
from ui.cluster_page import show_cluster_analysis

st.set_page_config(
    page_title="以太坊交易行为分析",
    page_icon="⛓️",
    layout="wide"
)

st.sidebar.title("⛓️ 以太坊交易分析")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "选择页面",
    ["数据概览", "欺诈检测", "异常检测", "聚类分析"]
)

# 加载数据（缓存，避免重复读取）
@st.cache_data
def load_and_process():
    df = load_data()
    df = engineer_features(df)
    return df

with st.spinner("正在加载数据..."):
    df = load_and_process()

X = get_feature_matrix(df)
y = get_labels(df)

st.sidebar.markdown("---")
st.sidebar.info(f"📊 数据集: {len(df)} 条记录\n🚨 欺诈: {int(y.sum())} ({y.mean():.1%})")

# 页面路由
if page == "数据概览":
    show_overview(df)

elif page == "欺诈检测":
    with st.spinner("正在训练欺诈检测模型..."):
        detector = FraudDetector()
        X_train, X_test, y_train, y_test = split_data(df)
        detector.train(X_train, y_train)
    show_fraud_detection(detector, X_test, y_test)

elif page == "异常检测":
    with st.spinner("正在训练异常检测模型..."):
        normal_df = df[df['FLAG'] == 0]
        X_normal = get_feature_matrix(normal_df)
        detector = AnomalyDetector()
        detector.fit(X_normal)
        anomaly_df = detector.predict(X, df['Address'])
    show_anomaly_detection(detector, anomaly_df)

elif page == "聚类分析":
    with st.spinner("正在进行聚类分析..."):
        analyzer = ClusterAnalyzer()
        analyzer.fit(X)
        cluster_df = analyzer.predict(X, df['Address'])
        # 合并FLAG到cluster_df用于统计
        cluster_with_flag = cluster_df.copy()
        cluster_with_flag['FLAG'] = y.values
        stats_df = analyzer.get_cluster_stats(cluster_with_flag, y)
    show_cluster_analysis(cluster_df, stats_df)
