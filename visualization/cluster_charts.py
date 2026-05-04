import plotly.graph_objects as go
import plotly.express as px


def plot_cluster_scatter(cluster_df):
    """聚类散点图（PCA降维到2D）"""
    fig = px.scatter(
        cluster_df,
        x='pca_x',
        y='pca_y',
        color='cluster',
        title='地址行为聚类（PCA降维）',
        color_discrete_sequence=px.colors.qualitative.Set1,
        hover_data=['address']
    )
    return fig


def plot_cluster_stats(stats_df):
    """聚类统计信息图"""
    fig = px.bar(
        stats_df,
        x='cluster',
        y='count',
        title='各聚类样本数量',
        color='fraud_ratio',
        color_continuous_scale='RdYlBu_r',
        text='fraud_ratio'
    )
    fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
    return fig
