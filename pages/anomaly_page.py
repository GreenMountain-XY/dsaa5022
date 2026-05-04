import streamlit as st
from visualization.anomaly_charts import plot_anomaly_score_distribution, plot_top_anomalies


def show_anomaly_detection(detector, anomaly_df):
    """异常检测页：展示异常散点图、可疑地址列表"""
    st.header("🔍 异常检测")
    
    # 异常统计
    st.subheader("异常统计")
    n_anomaly = anomaly_df['is_anomaly'].sum()
    n_total = len(anomaly_df)
    col1, col2, col3 = st.columns(3)
    col1.metric("总检测地址", n_total)
    col2.metric("检测异常数", int(n_anomaly))
    col3.metric("异常比例", f"{n_anomaly/n_total:.1%}")
    
    # 异常分数分布
    st.subheader("异常分数分布")
    st.plotly_chart(plot_anomaly_score_distribution(anomaly_df), use_container_width=True)
    
    # 最可疑地址
    st.subheader("最可疑地址 TOP 20")
    top_anomalies = detector.get_top_anomalies(anomaly_df, n=20)
    st.plotly_chart(plot_top_anomalies(top_anomalies), use_container_width=True)
    
    # 详细表格
    st.subheader("可疑地址详情")
    st.dataframe(top_anomalies)
