from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import NotFittedError


class AnomalyDetector:
    def __init__(self, contamination: float = 0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scores: Optional[pd.Series] = None
        self.last_predictions: Optional[pd.DataFrame] = None
        self.feature_columns: Optional[list[str]] = None
        self._is_fitted = False

    def fit(self, X_normal: pd.DataFrame) -> None:
        """
        Train the anomaly detector on normal addresses only.
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
        Detect anomalies and return address-level scores and labels.
        Returns: DataFrame(columns=['address', 'anomaly_score', 'is_anomaly'])
        """
        if not self._is_fitted:
            raise NotFittedError("Call fit() before predict().")

        features = self._prepare_features(X, fit=False)
        address_series = pd.Series(addresses).reset_index(drop=True)
        if len(features) != len(address_series):
            raise ValueError("X and addresses must have the same number of rows.")

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
        Return the n most suspicious addresses ranked by anomaly score.
        Returns: DataFrame(columns=['address', 'anomaly_score'])
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
