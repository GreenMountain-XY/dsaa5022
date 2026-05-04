import plotly.graph_objects as go
import plotly.express as px


def plot_feature_importance(importance_df):
    """特征重要性条形图"""
    fig = px.bar(
        importance_df.head(15),
        x='importance',
        y='feature',
        orientation='h',
        title='Top 15 特征重要性（Random Forest）',
        color='importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig


def plot_confusion_matrix(y_true, y_pred):
    """混淆矩阵热力图"""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="预测标签", y="真实标签", color="数量"),
        x=['正常(0)', '欺诈(1)'],
        y=['正常(0)', '欺诈(1)'],
        title='混淆矩阵',
        color_continuous_scale='Blues'
    )
    return fig


def plot_fraud_distribution(df, feature='total transactions (including tnx to create contract'):
    """欺诈 vs 正常分布对比图"""
    fig = px.histogram(
        df,
        x=feature,
        color='FLAG',
        title=f'{feature} 分布：欺诈 vs 正常',
        barmode='overlay',
        opacity=0.7,
        color_discrete_map={0: '#3498db', 1: '#e74c3c'}
    )
    fig.update_layout(legend_title_text='FLAG')
    return fig
