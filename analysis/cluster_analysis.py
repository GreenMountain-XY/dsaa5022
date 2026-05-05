import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler


class ClusterAnalyzer:
    """
    聚类分析器：对地址进行 K-Means 聚类，并用 PCA 降维到2D可视化
    """
    
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.pca = PCA(n_components=2, random_state=random_state)
        self.scaler = RobustScaler()
        self.random_state = random_state
        self._cluster_labels = None  # 存储最近一次 predict 的聚类结果
    
    def fit(self, X: pd.DataFrame) -> None:
        """训练聚类模型（自动标准化）"""
        X_scaled = self.scaler.fit_transform(X.fillna(0))
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = np.clip(X_scaled, -5, 5)
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.kmeans.fit(X_scaled)
    
    def predict(self, X: pd.DataFrame, addresses: pd.Series) -> pd.DataFrame:
        """
        聚类并降维
        Returns: DataFrame(columns=['address', 'cluster', 'pca_x', 'pca_y'])
        """
        X_scaled = self.scaler.transform(X.fillna(0))
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = np.clip(X_scaled, -5, 5)
        clusters = self.kmeans.predict(X_scaled)
        pca_result = self.pca.fit_transform(X_scaled)
        
        self._cluster_labels = clusters  # 保存供 get_cluster_stats 使用
        
        result = pd.DataFrame({
            'address': addresses.values,
            'cluster': clusters,
            'pca_x': pca_result[:, 0],
            'pca_y': pca_result[:, 1]
        })
        return result
    
    def get_cluster_stats(self, df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        """
        获取每个聚类的统计信息
        Returns: DataFrame(columns=['cluster', 'count', 'fraud_ratio', ...])
        """
        if self._cluster_labels is None:
            raise RuntimeError("请先调用 predict() 再调用 get_cluster_stats()")
        
        df_copy = df.copy().reset_index(drop=True)
        df_copy['cluster'] = self._cluster_labels
        
        stats = []
        for c in sorted(df_copy['cluster'].unique()):
            subset = df_copy[df_copy['cluster'] == c]
            stats.append({
                'cluster': c,
                'count': len(subset),
                'fraud_ratio': subset['FLAG'].mean() if 'FLAG' in subset.columns else 0,
                'avg_total_transactions': subset['total transactions (including tnx to create contract'].mean() if 'total transactions (including tnx to create contract' in subset.columns else 0,
                'avg_ether_sent': subset['total Ether sent'].mean() if 'total Ether sent' in subset.columns else 0
            })
        return pd.DataFrame(stats)
