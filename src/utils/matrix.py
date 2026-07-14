"""矩阵通用变换工具。"""

import numpy as np
from numpy import ndarray


def positive_transform(
    matrix: ndarray,
    kinds: list[int] | tuple[int, ...],
    best_values: list[float | None] | None = None,
    intervals: list[tuple[float, float] | None] | None = None,
) -> ndarray:
    """将指标矩阵统一转化为极大型指标。

    Args:
        matrix: 原始评价矩阵，形状为 (n, m)。
        kinds: 每列的指标类型，取值：
            1 - 极大型（无需转换）
            2 - 极小型
            3 - 中间型
            4 - 区间型
        best_values: 中间型列对应的最优值，其他列可为 None。
        intervals: 区间型列对应的 (下限, 上限)，其他列可为 None。

    Returns:
        转化为极大型指标后的矩阵。

    Raises:
        ValueError: 参数长度不匹配、缺少必要参数、区间下限大于上限，
            或不支持的指标类型。
    """
    if matrix.ndim != 2:
        raise ValueError("matrix 必须是二维数组")

    n, m = matrix.shape
    kinds = list(kinds)
    if len(kinds) != m:
        raise ValueError(f"kinds 长度 {len(kinds)} 与矩阵列数 {m} 不一致")

    if best_values is None:
        best_values = [None] * m
    if intervals is None:
        intervals = [None] * m

    if len(best_values) != m:
        raise ValueError(f"best_values 长度 {len(best_values)} 与矩阵列数 {m} 不一致")
    if len(intervals) != m:
        raise ValueError(f"intervals 长度 {len(intervals)} 与矩阵列数 {m} 不一致")

    converted = matrix.copy()

    for j, kind in enumerate(kinds):
        col = matrix[:, j]

        if kind == 1:
            continue

        if kind == 2:
            converted[:, j] = np.max(col) - col
            continue

        if kind == 3:
            best = best_values[j]
            if best is None:
                raise ValueError(f"第 {j + 1} 列为中间型，需要提供 best_values")
            max_diff = np.max(np.abs(col - best))
            if max_diff == 0:
                converted[:, j] = 1.0
            else:
                converted[:, j] = 1 - np.abs(col - best) / max_diff
            continue

        if kind == 4:
            interval = intervals[j]
            if interval is None:
                raise ValueError(f"第 {j + 1} 列为区间型，需要提供 intervals")
            a, b = interval
            if b < a:
                raise ValueError(f"第 {j + 1} 列区间下限 {a} 不能大于上限 {b}")
            M = max(a - np.min(col), np.max(col) - b)
            if M == 0:
                converted[:, j] = 1.0
            else:
                new_col = np.ones(n)
                below = col < a
                above = col > b
                new_col[below] = 1 - (a - col[below]) / M
                new_col[above] = 1 - (col[above] - b) / M
                converted[:, j] = new_col
            continue

        raise ValueError(f"不支持的指标类型：{kind}")

    return converted


def vector_normalize(matrix: ndarray) -> ndarray:
    """标准化：每列除以其欧几里得范数，使列向量长度为 1。

    Args:
        matrix: 待标准化矩阵。

    Returns:
        标准化后的矩阵。

    Raises:
        ValueError: 存在全零列时无法标准化。
    """
    norms = np.linalg.norm(matrix, axis=0)
    if np.any(norms == 0):
        raise ValueError("存在全零列，无法进行向量归一化。")
    return matrix / norms


def sum_normalize(matrix: ndarray) -> ndarray:
    """归一化：每列除以其列和，使每列之和为 1。

    Args:
        matrix: 待归一化矩阵。

    Returns:
        归一化后的矩阵。

    Raises:
        ValueError: 存在列和为 0 的列时无法归一化。
    """
    col_sum = np.sum(matrix, axis=0)
    if np.any(col_sum == 0):
        raise ValueError("存在列和为 0 的列，无法进行归一化。")
    return matrix / col_sum
