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
from src.utils.matrix import positive_transform, vector_normalize


def convert_indicators(
    matrix: ndarray,
    kinds: list[int],
    best_values: list[float | None] | None = None,
    intervals: list[tuple[float, float] | None] | None = None,
) -> ndarray:
    """将各类指标统一转化为极大型指标。

    kinds 取值：
        1 - 极大型（无需转换）
        2 - 极小型
        3 - 中间型
        4 - 区间型

    这是 ``src.utils.matrix.positive_transform`` 的薄包装，
    保留原名以保持现有接口兼容。
    """
    return positive_transform(matrix, kinds, best_values, intervals)


def normalize_matrix(matrix: ndarray) -> ndarray:
    """向量归一化：每列除以其欧几里得范数。

    这是 ``src.utils.matrix.vector_normalize`` 的薄包装。
    """
    return vector_normalize(matrix)


def weighted_normalized_matrix(normalized: ndarray, weights: ndarray) -> ndarray:
    """构造加权规范化矩阵。"""
    return normalized * weights


def ideal_solutions(weighted: ndarray) -> tuple[ndarray, ndarray]:
    """计算正理想解（每列最大值）和负理想解（每列最小值）。

    要求所有指标已统一为极大型指标。
    """
    v_pos = np.max(weighted, axis=0)
    v_neg = np.min(weighted, axis=0)
    return v_pos, v_neg


def calculate_closeness(
    weighted: ndarray, v_pos: ndarray, v_neg: ndarray
) -> ndarray:
    """计算每个方案与理想解的相对贴近度 C_i。"""
    d_pos = np.linalg.norm(weighted - v_pos, axis=1)
    d_neg = np.linalg.norm(weighted - v_neg, axis=1)
    denominator = d_pos + d_neg
    if np.any(denominator == 0):
        raise ValueError("某方案同时等于正理想解与负理想解，贴近度无定义。")
    return d_neg / denominator


def read_weights(m: int) -> ndarray:
    """交互式读取指标权重；若用户不输入则采用等权重。"""
    print("\n请输入各指标权重（用空格分隔），直接回车则使用等权重：")
    while True:
        line = input(">>> 权重：").strip()
        if line == "":
            weights = np.ones(m) / m
            print(f"使用等权重：{np.round(weights, 4).tolist()}")
            return weights
        try:
            weights = np.array([float(x) for x in line.split()])
        except ValueError:
            print("请输入数字，并用空格分隔。")
            continue
        if len(weights) != m:
            print(f"需要输入 {m} 个权重，实际输入 {len(weights)} 个。")
            continue
        if np.any(weights < 0):
            print("权重不能为负数。")
            continue
        if np.isclose(np.sum(weights), 0):
            print("权重总和不能为 0。")
            continue
        return weights / np.sum(weights)


def interactive_topsis():
    """控制台交互式 TOPSIS 入口。"""
    print("=== TOPSIS 分析法 ===")
    n = read_int("请输入参评数目：", min_value=1)
    m = read_int("请输入指标数目：", min_value=1)

    print("\n指标类型说明：1=极大型，2=极小型，3=中间型，4=区间型")
    kinds = read_ints(
        "请输入类型矩阵（用空格分隔）：",
        count=m,
        positive=False,
    )
    # 校验类型取值范围
    for k in kinds:
        if k not in (1, 2, 3, 4):
            raise ValueError(f"指标类型必须是 1/2/3/4 之一，得到 {k}")

    best_values, intervals = read_indicator_params(kinds)

    print("\n请输入评价矩阵：")
    matrix = read_matrix(n, m)

    print("\n步骤 1：统一转化为极大型指标...")
    converted = convert_indicators(matrix, kinds, best_values, intervals)
    print_matrix(converted, title="转化后的极大型指标矩阵：")

    print("\n步骤 2：向量归一化...")
    normalized = normalize_matrix(converted)
    print_matrix(normalized, title="规范化矩阵：")

    weights = read_weights(m)

    print("\n步骤 3：构造加权规范化矩阵...")
    weighted = weighted_normalized_matrix(normalized, weights)
    print_matrix(weighted, title="加权规范化矩阵：")

    print("\n步骤 4：确定正理想解与负理想解...")
    v_pos, v_neg = ideal_solutions(weighted)
    print(f"正理想解 V+：{np.round(v_pos, 4).tolist()}")
    print(f"负理想解 V-：{np.round(v_neg, 4).tolist()}")

    print("\n步骤 5：计算贴近度并排序...")
    closeness = calculate_closeness(weighted, v_pos, v_neg)
    ranks = np.argsort(-closeness) + 1  # 贴近度越大越优

    print("\n=== 评价结果 ===")
    print(f"{'方案':<6}{'贴近度 C':<12}{'排名':<6}")
    for i in range(n):
        print(f"{i + 1:<6}{closeness[i]:<12.4f}{np.where(ranks == i + 1)[0][0] + 1:<6}")

    best = int(ranks[0])
    print(f"\n最优方案为：方案 {best}")


if __name__ == "__main__":
    interactive_topsis()
