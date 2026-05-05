from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_anomaly_scatter(df: pd.DataFrame) -> go.Figure:
    """
    异常检测散点图。

    横轴是样本序号，纵轴是异常分数，颜色区分正常/异常。
    这个版本不依赖 PCA，方便直接接文档里定义的 `anomaly_df`。
    """
    required_columns = {"address", "anomaly_score", "is_anomaly"}
    if df is None or df.empty:
        return _empty_figure("异常检测散点图", "暂无异常检测结果。")
    if not required_columns.issubset(df.columns):
        missing = ", ".join(sorted(required_columns - set(df.columns)))
        raise ValueError(f"plot_anomaly_scatter missing required columns: {missing}")

    plot_df = df.reset_index(drop=True).copy()
    plot_df["sample_index"] = np.arange(len(plot_df))
    plot_df["label"] = plot_df["is_anomaly"].map({0: "正常", 1: "异常"}).fillna("未知")

    fig = go.Figure()
    for label, color in (("正常", "#4C78A8"), ("异常", "#E45756")):
        group = plot_df[plot_df["label"] == label]
        if group.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=group["sample_index"],
                y=group["anomaly_score"],
                mode="markers",
                name=label,
                marker={"size": 8, "opacity": 0.75, "color": color},
                customdata=group[["address"]],
                hovertemplate=(
                    "样本序号: %{x}<br>"
                    "异常分数: %{y:.4f}<br>"
                    "地址: %{customdata[0]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="异常检测散点图",
        xaxis_title="样本序号",
        yaxis_title="异常分数",
        legend_title="判定结果",
        template="plotly_white",
    )
    return fig


def plot_anomaly_score_distribution(scores: pd.Series) -> go.Figure:
    """异常分数分布直方图。"""
    score_series = pd.Series(scores).dropna()
    if score_series.empty:
        return _empty_figure("异常分数分布", "暂无异常分数可展示。")

    # 使用样本量的平方根来估算柱子数量，避免箱数过多或过少。
    nbins = max(10, min(40, int(np.sqrt(len(score_series)))))
    fig = go.Figure(
        data=[
            go.Histogram(
                x=score_series,
                nbinsx=nbins,
                marker={"color": "#72B7B2", "line": {"color": "#4C78A8", "width": 1}},
                opacity=0.85,
            )
        ]
    )
    fig.update_layout(
        title="异常分数分布",
        xaxis_title="异常分数",
        yaxis_title="样本数",
        template="plotly_white",
        bargap=0.08,
    )
    return fig


def plot_top_anomalies(top_df: pd.DataFrame) -> go.Figure:
    """Top N 可疑地址条形图。"""
    required_columns = {"address", "anomaly_score"}
    if top_df is None or top_df.empty:
        return _empty_figure("Top 可疑地址", "暂无可疑地址。")
    if not required_columns.issubset(top_df.columns):
        missing = ", ".join(sorted(required_columns - set(top_df.columns)))
        raise ValueError(f"plot_top_anomalies missing required columns: {missing}")

    plot_df = top_df.sort_values("anomaly_score", ascending=False).copy()
    plot_df["display_address"] = plot_df["address"].map(_shorten_address)

    fig = go.Figure(
        data=[
            go.Bar(
                x=plot_df["anomaly_score"],
                y=plot_df["display_address"],
                orientation="h",
                marker={"color": "#F58518"},
                customdata=plot_df[["address"]],
                hovertemplate=(
                    "地址: %{customdata[0]}<br>"
                    "异常分数: %{x:.4f}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title="Top 可疑地址",
        xaxis_title="异常分数",
        yaxis_title="地址",
        template="plotly_white",
        yaxis={"autorange": "reversed"},
    )
    return fig


def _empty_figure(title: str, message: str) -> go.Figure:
    """统一的空图占位，避免页面在无数据时直接报错。"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14},
    )
    fig.update_layout(title=title, template="plotly_white")
    return fig


def _shorten_address(address: str) -> str:
    """在图表 y 轴上缩短地址显示，避免标签过长。"""
    address = str(address)
    if len(address) <= 18:
        return address
    return f"{address[:8]}...{address[-6:]}"
