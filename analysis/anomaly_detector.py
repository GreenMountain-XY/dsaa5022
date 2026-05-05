from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import NotFittedError


class AnomalyDetector:
    """
    基于 Isolation Forest 的异常检测器。

    约定：
    1. `fit()` 只接收正常地址的特征矩阵；
    2. `predict()` 对全量地址打分；
    3. `anomaly_score` 越大，表示地址越可疑。
    """

    def __init__(self, contamination: float = 0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scores: Optional[pd.Series] = None
        self.last_predictions: Optional[pd.DataFrame] = None
        self.feature_columns: Optional[list[str]] = None
        self._is_fitted = False

    def fit(self, X_normal: pd.DataFrame) -> None:
        """
        在正常地址样本上训练异常检测模型。

        参数：
        - X_normal: 仅包含正常地址的特征矩阵，通常来自 `FLAG == 0` 的子集。
        """
        features = self._prepare_features(X_normal, fit=True)
        if features.empty:
            raise ValueError("X_normal must contain at least one numeric feature column.")

        self.model.fit(features)
        self.scores = None
        self.last_predictions = None
        self._is_fitted = True

    def predict(self, X: pd.DataFrame, addresses: pd.Series) -> pd.DataFrame:
        """
        对输入地址进行异常检测，返回地址级别的打分结果。

        返回：
        - DataFrame(columns=['address', 'anomaly_score', 'is_anomaly'])
        - `is_anomaly`: 1 表示异常，0 表示正常
        """
        if not self._is_fitted:
            raise NotFittedError("Call fit() before predict().")

        features = self._prepare_features(X, fit=False)
        address_series = pd.Series(addresses).reset_index(drop=True)
        if len(features) != len(address_series):
            raise ValueError("X and addresses must have the same number of rows.")

        # sklearn 的 decision_function 越小越异常，这里取负号，
        # 统一成“分数越大越可疑”，便于页面展示和排序。
        anomaly_scores = -self.model.decision_function(features)
        raw_predictions = self.model.predict(features)
        is_anomaly = (raw_predictions == -1).astype(int)

        result = pd.DataFrame(
            {
                "address": address_series.astype(str),
                "anomaly_score": anomaly_scores,
                "is_anomaly": is_anomaly,
            }
        )

        self.scores = result["anomaly_score"].copy()
        self.last_predictions = result.copy()
        return result

    def get_top_anomalies(self, n: int = 20) -> pd.DataFrame:
        """
        返回异常分数最高的前 n 个可疑地址。

        返回：
        - DataFrame(columns=['address', 'anomaly_score'])
        """
        if self.last_predictions is None:
            raise ValueError("Call predict() before get_top_anomalies().")

        top_n = max(int(n), 0)
        top_df = (
            self.last_predictions.sort_values("anomaly_score", ascending=False)
            .head(top_n)
            .loc[:, ["address", "anomaly_score"]]
            .reset_index(drop=True)
        )
        return top_df

    def _prepare_features(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        清洗并对齐模型输入特征。

        训练时记录列顺序；预测时严格按训练列顺序取列，避免
        因为列缺失或顺序变化导致模型输入不一致。
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        numeric_df = X.select_dtypes(include=[np.number]).copy()
        if numeric_df.empty:
            raise ValueError("Feature matrix must contain numeric columns.")

        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        if fit:
            self.feature_columns = list(numeric_df.columns)
            return numeric_df.reset_index(drop=True)

        if self.feature_columns is None:
            raise NotFittedError("Call fit() before predict().")

        missing_columns = [col for col in self.feature_columns if col not in numeric_df.columns]
        if missing_columns:
            raise ValueError(
                "Predict feature matrix is missing columns: "
                + ", ".join(missing_columns)
            )

        return numeric_df.loc[:, self.feature_columns].reset_index(drop=True)
