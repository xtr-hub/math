"""熵权法实现。

基于信息熵计算各指标客观权重：信息量越大（差异越大）的指标权重越高。
"""

# ruff: noqa: E402
import sys
from pathlib import Path

import numpy as np
from numpy import ndarray

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.io.matrix_io import (
    print_matrix,
    read_indicator_params,
    read_int,
    read_ints,
    read_matrix,
)
from src.utils.matrix import positive_transform, sum_normalize


def calculate_entropy_weights(
    matrix: ndarray,
    kinds: list[int] | None = None,
    best_values: list[float | None] | None = None,
    intervals: list[tuple[float, float] | None] | None = None,
    epsilon: float = 1e-10,
) -> ndarray:
    """使用熵权法计算指标权重。

    Args:
        matrix: 原始评价矩阵，形状为 (n, m)。
        kinds: 可选，每列指标类型。若提供，则先进行正向化。
            取值：1=极大型，2=极小型，3=中间型，4=区间型。
        best_values: 中间型列对应的最优值，其他列可为 None。
        intervals: 区间型列对应的 (下限, 上限)，其他列可为 None。
        epsilon: 用于判断零值的小常数。

    Returns:
        归一化权重向量，形状为 (m,)，所有元素非负且和为 1。

    Raises:
        ValueError: 矩阵包含负数或参数不合法。
    """
    if matrix.ndim != 2:
        raise ValueError("matrix 必须是二维数组")

    n, m = matrix.shape
    if n == 0 or m == 0:
        raise ValueError("matrix 不能为空")

    # 只有单个方案时无法区分信息熵，返回等权重
    if n == 1:
        return np.ones(m) / m

    data = matrix.copy()
    if kinds is not None:
        data = positive_transform(data, kinds, best_values, intervals)

    if np.any(data < -epsilon):
        raise ValueError("熵权法要求矩阵元素非负，请先进行正向化或检查数据")

    # 负值在 epsilon 范围内视为 0，避免浮点误差
    data = np.where(data < 0, 0.0, data)

    col_sum = np.sum(data, axis=0)
    p = np.zeros_like(data, dtype=float)

    valid_cols = col_sum > epsilon
    if np.any(valid_cols):
        p[:, valid_cols] = data[:, valid_cols] / col_sum[valid_cols]

    # 避免 log(0)
    k = 1.0 / np.log(n)
    p_safe = np.where(p > epsilon, p, 1.0)
    entropy = -k * np.sum(p * np.log(p_safe), axis=0)

    # 列和为 0 的列没有信息量，熵值取 1
    entropy[~valid_cols] = 1.0

    redundancy = 1.0 - entropy
    redundancy_sum = np.sum(redundancy)

    if redundancy_sum <= epsilon:
        return np.ones(m) / m

    return redundancy / redundancy_sum


def interactive_entropy_weight():
    """控制台交互式熵权法入口。"""
    print("=== 熵权法 ===")
    n = read_int("请输入参评数目：", min_value=1)
    m = read_int("请输入指标数目：", min_value=1)

    print("\n指标类型说明：1=极大型，2=极小型，3=中间型，4=区间型")
    print("若全部为极大型指标，可直接输入全 1。")
    kinds = read_ints(
        "请输入类型矩阵（用空格分隔）：",
        count=m,
        positive=False,
    )
    for k in kinds:
        if k not in (1, 2, 3, 4):
            raise ValueError(f"指标类型必须是 1/2/3/4 之一，得到 {k}")

    best_values, intervals = read_indicator_params(kinds)

    print("\n请输入评价矩阵：")
    matrix = read_matrix(n, m)

    print("\n步骤 1：统一转化为极大型指标...")
    converted = positive_transform(matrix, kinds, best_values, intervals)
    print_matrix(converted, title="正向化后的矩阵：")

    print("\n步骤 2：列和归一化得到比重矩阵...")
    normalized = sum_normalize(converted)
    print_matrix(normalized, title="比重矩阵 P：")

    print("\n步骤 3：计算熵权...")
    weights = calculate_entropy_weights(
        matrix,
        kinds=kinds,
        best_values=best_values,
        intervals=intervals,
    )
    print(f"\n指标权重：{np.round(weights, 4).tolist()}")


if __name__ == "__main__":
    interactive_entropy_weight()
