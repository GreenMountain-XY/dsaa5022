import pandas as pd
import numpy as np
import os


def load_data(path='data/ethereum_fraud.csv') -> pd.DataFrame:
    """
    加载以太坊欺诈检测数据集
    Returns: DataFrame，包含所有原始列
    """
    df = pd.read_csv(path)
    # 去除空列名或Unnamed列
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # strip列名前后空格，统一格式
    df.columns = df.columns.str.strip()
    return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    获取特征矩阵（去掉 Address、FLAG、Index 等非特征列）
    Input: load_data() 输出的 DataFrame
    Returns: X (n_samples, n_features)
    """
    drop_cols = ['Address', 'FLAG']
    # 如果有 Index 列也去掉
    if 'Index' in df.columns:
        drop_cols.append('Index')
    # 去掉文本列（token类型）
    text_cols = ['ERC20 most sent token type', 'ERC20_most_rec_token_type']
    for col in text_cols:
        if col in df.columns:
            drop_cols.append(col)
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    return X


def get_labels(df: pd.DataFrame) -> pd.Series:
    """
    获取标签列
    Returns: y (n_samples,)
    """
    return df['FLAG']
