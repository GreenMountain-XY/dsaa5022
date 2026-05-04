import streamlit as st
from visualization.fraud_charts import plot_feature_importance, plot_confusion_matrix, plot_fraud_distribution


def show_fraud_detection(detector, X_test, y_test):
    """欺诈检测页：展示模型性能、特征重要性、分布对比"""
    st.header("🚨 欺诈检测")
    
    # 模型评估
    st.subheader("模型性能")
    metrics = detector.evaluate(X_test, y_test)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("准确率", f"{metrics['accuracy']:.2%}")
    col2.metric("精确率", f"{metrics['precision']:.2%}")
    col3.metric("召回率", f"{metrics['recall']:.2%}")
    col4.metric("F1分数", f"{metrics['f1']:.2%}")
    
    # 特征重要性
    st.subheader("特征重要性")
    importance_df = detector.get_feature_importance()
    st.plotly_chart(plot_feature_importance(importance_df), use_container_width=True)
    
    # 混淆矩阵
    st.subheader("混淆矩阵")
    y_pred = detector.predict(X_test)
    st.plotly_chart(plot_confusion_matrix(y_test, y_pred), use_container_width=True)
    
    # 分布对比
    st.subheader("欺诈 vs 正常 分布对比")
    # 这里需要传入完整DataFrame，由app.py传入
    # st.plotly_chart(plot_fraud_distribution(df), use_container_width=True)
    st.info("分布对比图：建议展示 total transactions 或 total Ether sent 的分布差异")
