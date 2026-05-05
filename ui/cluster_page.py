import streamlit as st
from visualization.cluster_charts import plot_cluster_scatter, plot_cluster_stats


def show_cluster_analysis(cluster_df, stats_df):
    """聚类分析页：展示聚类散点图、聚类统计"""
    st.header("🎯 聚类分析")
    
    # 聚类散点图
    st.subheader("地址行为聚类（PCA降维）")
    st.plotly_chart(plot_cluster_scatter(cluster_df), use_container_width=True)
    
    # 聚类统计
    st.subheader("各聚类统计")
    st.plotly_chart(plot_cluster_stats(stats_df), use_container_width=True)
    
    # 统计表格
    st.subheader("聚类详细信息")
    st.dataframe(stats_df)
