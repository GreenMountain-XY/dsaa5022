import streamlit as st


def show_overview(df):
    """数据概览页：展示统计卡片、数据表格、基本分布"""
    st.header("📊 数据概览")
    
    # 统计卡片
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总地址数", len(df))
    col2.metric("欺诈地址", int(df['FLAG'].sum()))
    col3.metric("正常地址", int(len(df) - df['FLAG'].sum()))
    col4.metric("欺诈比例", f"{df['FLAG'].mean():.1%}")
    
    # 数据表格
    st.subheader("数据集样本")
    st.dataframe(df.head(20))
    
    # 基本分布
    st.subheader("基本统计")
    st.write(df.describe())
