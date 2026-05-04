import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class ClusterAnalyzer:
    """
    聚类分析器：对地址进行 K-Means 聚类，并用 PCA 降维到2D可视化
    """
    
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.pca = PCA(n_components=2, random_state=random_state)
        self.random_state = random_state
    
    def fit(self, X: pd.DataFrame) -> None:
        """训练聚类模型"""
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.kmeans.fit(X)
    
    def predict(self, X: pd.DataFrame, addresses: pd.Series) -> pd.DataFrame:
        """
        聚类并降维
        Returns: DataFrame(columns=['address', 'cluster', 'pca_x', 'pca_y'])
        """
        clusters = self.kmeans.predict(X)
        pca_result = self.pca.fit_transform(X)
        
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
        # 需要 cluster 列已在 df 中
        stats = []
        for c in sorted(df['cluster'].unique()):
            subset = df[df['cluster'] == c]
            stats.append({
                'cluster': c,
                'count': len(subset),
                'fraud_ratio': subset['FLAG'].mean() if 'FLAG' in subset.columns else 0,
                'avg_total_transactions': subset['total transactions (including tnx to create contract'].mean() if 'total transactions (including tnx to create contract' in subset.columns else 0,
                'avg_ether_sent': subset['total Ether sent'].mean() if 'total Ether sent' in subset.columns else 0
            })
        return pd.DataFrame(stats)
