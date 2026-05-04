import plotly.graph_objects as go
import plotly.express as px


def plot_anomaly_scatter(anomaly_df, df):
    """
    异常检测散点图
    anomaly_df: columns=['address', 'anomaly_score', 'is_anomaly']
    df: 原始DataFrame（用于获取额外特征做颜色映射）
    """
    fig = px.scatter(
        anomaly_df,
        x='anomaly_score',
        y=anomaly_df.index,
        color='is_anomaly',
        title='异常检测分数分布',
        color_discrete_map={0: '#3498db', 1: '#e74c3c'},
        labels={'is_anomaly': '是否异常', 'anomaly_score': '异常分数'}
    )
    return fig


def plot_anomaly_score_distribution(scores):
    """异常分数分布直方图"""
    fig = px.histogram(
        scores,
        x='anomaly_score',
        color='is_anomaly',
        title='异常分数分布',
        barmode='overlay',
        color_discrete_map={0: '#3498db', 1: '#e74c3c'}
    )
    return fig


def plot_top_anomalies(top_df):
    """最可疑地址排名条形图"""
    fig = px.bar(
        top_df,
        x='anomaly_score',
        y='address',
        orientation='h',
        title=f'最可疑的 {len(top_df)} 个地址',
        color='anomaly_score',
        color_continuous_scale='Reds'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig
