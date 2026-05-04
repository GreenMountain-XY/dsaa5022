import pandas as pd
from sklearn.model_selection import train_test_split


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    特征工程：添加衍生特征
    Input: 原始DataFrame（包含load_data的全部列）
    Returns: 添加衍生列后的DataFrame
    """
    df = df.copy()
    
    # 避免除以0
    eps = 1e-6
    
    # 基础比率特征
    df['sent_received_ratio'] = df['Sent tnx'] / (df['Received Tnx'] + eps)
    df['avg_transaction_value'] = df['total Ether sent'] / (df['Sent tnx'] + eps)
    df['contract_ratio'] = df['Number of Created Contracts'] / (df['total transactions (including tnx to create contract'] + eps)
    df['balance_per_transaction'] = df['total ether balance'] / (df['total transactions (including tnx to create contract'] + eps)
    
    # 时间相关特征
    df['total_time_active'] = df['Time Diff between first and last (Mins)']
    df['avg_sent_interval'] = df['Avg min between sent tnx']
    df['avg_received_interval'] = df['Avg min between received tnx']
    
    # ETH流动特征
    df['ether_flow'] = df['total ether received'] - df['total Ether sent']
    df['max_received_ratio'] = df['max value received '] / (df['avg val received'] + eps)
    
    # ERC20活跃度
    df['erc20_activity'] = df[' Total ERC20 tnxs'] / (df['total transactions (including tnx to create contract'] + eps)
    
    # 填充任何NaN（防止模型报错）
    df = df.fillna(0)
    
    return df


def split_data(df: pd.DataFrame, test_size=0.2, random_state=42) -> tuple:
    """
    划分训练/测试集
    注意：使用engineer_features之后的DataFrame
    Returns: (X_train, X_test, y_train, y_test)
    """
    from data_module.loader import get_feature_matrix, get_labels
    
    X = get_feature_matrix(df)
    y = get_labels(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
