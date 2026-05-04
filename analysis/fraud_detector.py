import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


class FraudDetector:
    """
    欺诈检测器：使用 Random Forest 对地址进行分类（欺诈 vs 正常）
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight='balanced'  # 处理类别不平衡
        )
        self.metrics = {}
        self.is_trained = False
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """训练分类器"""
        self.model.fit(X_train, y_train)
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测标签"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        评估模型
        Returns: {'accuracy': float, 'precision': float, 'recall': float, 'f1': float}
        """
        y_pred = self.predict(X_test)
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        return self.metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性
        Returns: DataFrame(columns=['feature', 'importance'])
        """
        importance = pd.DataFrame({
            'feature': self.model.feature_names_in_,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        return importance
