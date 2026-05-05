from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from analysis.anomaly_detector import AnomalyDetector
from visualization.anomaly_charts import (
    plot_anomaly_scatter,
    plot_anomaly_score_distribution,
    plot_top_anomalies,
)


def show_anomaly_detection(detector: AnomalyDetector, anomaly_df: pd.DataFrame):
    """
    Streamlit 异常检测页。

    输入：
    - detector: 已完成 fit/predict 的异常检测器
    - anomaly_df: `predict()` 返回的结果表
    """
    st.title("异常检测")
    st.caption("基于 Isolation Forest 的无监督异常检测，异常分数越高越可疑。")

    if anomaly_df is None or anomaly_df.empty:
        st.info("暂无异常检测结果。请先完成模型训练和预测。")
        return

    required_columns = {"address", "anomaly_score", "is_anomaly"}
    if not required_columns.issubset(anomaly_df.columns):
        missing = ", ".join(sorted(required_columns - set(anomaly_df.columns)))
        st.error(f"异常检测结果缺少必要列: {missing}")
        return

    result_df = anomaly_df.reset_index(drop=True).copy()
    # 页面顶部先给出最核心的 4 个指标，方便演示时快速讲清结果。
    total_count = len(result_df)
    anomaly_count = int(result_df["is_anomaly"].sum())
    anomaly_ratio = anomaly_count / total_count if total_count else 0.0
    highest_score_row = result_df.sort_values("anomaly_score", ascending=False).iloc[0]
    mean_score = float(result_df["anomaly_score"].mean()) if total_count else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("样本总数", f"{total_count}")
    col2.metric("异常地址数", f"{anomaly_count}")
    col3.metric("异常占比", f"{anomaly_ratio:.2%}")
    col4.metric("平均异常分数", f"{mean_score:.4f}")

    st.metric("最高分地址", str(highest_score_row["address"]))

    st.plotly_chart(plot_anomaly_scatter(result_df), width="stretch")

    # 正常情况下优先使用 detector 内保存的完整分数；
    # 如果没有，则退回当前结果表中的 anomaly_score。
    score_series = (
        detector.scores
        if detector.scores is not None
        else result_df["anomaly_score"]
    )
    st.plotly_chart(
        plot_anomaly_score_distribution(score_series),
        width="stretch",
    )

    st.subheader("Top 可疑地址")
    top_df = _resolve_top_anomalies(detector, result_df, n=20)
    st.plotly_chart(plot_top_anomalies(top_df), width="stretch")
    st.dataframe(top_df, width="stretch", hide_index=True)


def _resolve_top_anomalies(
    detector: AnomalyDetector, anomaly_df: pd.DataFrame, n: int
) -> pd.DataFrame:
    """
    统一获取 Top 异常地址。

    优先调用 detector 内部缓存的预测结果；如果缓存缺失，
    就直接从传入的 anomaly_df 里兜底计算。
    """
    try:
        return detector.get_top_anomalies(n=n)
    except ValueError:
        return (
            anomaly_df.sort_values("anomaly_score", ascending=False)
            .head(max(int(n), 0))
            .loc[:, ["address", "anomaly_score"]]
            .reset_index(drop=True)
        )


if __name__ == "__main__":
    # 这段演示数据用于单独运行页面文件时快速预览布局。
    demo_df = pd.DataFrame(
        {
            "address": [
                "0x1111111111111111111111111111111111111111",
                "0x2222222222222222222222222222222222222222",
                "0x3333333333333333333333333333333333333333",
                "0x4444444444444444444444444444444444444444",
                "0x5555555555555555555555555555555555555555",
            ],
            "anomaly_score": [0.12, 0.88, 0.41, 1.26, 0.19],
            "is_anomaly": [0, 1, 0, 1, 0],
        }
    )
    demo_detector = AnomalyDetector()
    demo_detector.scores = demo_df["anomaly_score"].copy()
    demo_detector.last_predictions = demo_df.copy()
    show_anomaly_detection(demo_detector, demo_df)
