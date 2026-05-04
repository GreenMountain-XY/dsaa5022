import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """
    异常检测器：使用 Isolation Forest 在 FLAG=0 的数据上训练，
    然后在全部数据上检测异常。
    """
    
    def __init__(self, contamination=0.05, random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.scores = None
        self.is_fitted = False
    
    def fit(self, X_normal: pd.DataFrame) -> None:
        """
        在 FLAG=0（正常地址）的数据上训练异常检测模型
        Input: 正常地址的特征矩阵（DataFrame）
        """
        self.model.fit(X_normal)
        self.is_fitted = True
    
    def predict(self, X: pd.DataFrame, addresses: pd.Series) -> pd.DataFrame:
        """
        对全部地址进行异常检测
        Returns: DataFrame(columns=['address', 'anomaly_score', 'is_anomaly'])
                 is_anomaly: 1=异常, 0=正常
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        scores = self.model.decision_function(X)
        predictions = self.model.predict(X)
        
        # predictions: -1=异常, 1=正常 → 转为 1=异常, 0=正常
        result = pd.DataFrame({
            'address': addresses.values,
            'anomaly_score': scores,
            'is_anomaly': (predictions == -1).astype(int)
        })
        return result
    
    def get_top_anomalies(self, anomaly_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        """
        获取最可疑的 n 个地址（anomaly_score 越低越异常）
        Returns: DataFrame(columns=['address', 'anomaly_score', 'is_anomaly'])
        """
        top = anomaly_df.nsmallest(n, 'anomaly_score')
        return top
