"""灰色关联分析 (Grey Relational Analysis, GRA) 实现。

基于灰色系统理论，通过比较各方案与参考序列（理想方案）的几何相似程度来评价方案的优劣。
关联度越大，方案越优。

算法步骤：
    1. 指标正向化（统一转化为极大型指标）
    2. 无量纲化处理（均值化 / 初值化）
    3. 确定参考序列（默认取各列最优值）
    4. 计算绝对差矩阵
    5. 计算灰色关联系数矩阵
    6. 计算灰色关联度并排序
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
from src.utils.matrix import positive_transform


def normalize_data(matrix: ndarray, method: str = "mean") -> ndarray:
    """对正向化后的矩阵进行无量纲化处理。

    支持两种方式：
        - ``"mean"``（均值化）：每列除以该列均值，消除量纲影响，保留变异信息。
        - ``"init"``（初值化）：每列除以该列第一个值，适用于时间序列等有序数据。

    Args:
        matrix: 已正向化的评价矩阵，形状为 ``(n, m)``。
        method: 无量纲化方法，``"mean"`` 或 ``"init"``。

    Returns:
        无量纲化后的矩阵，形状为 ``(n, m)``。

    Raises:
        ValueError: 列均值为零或初值为零时。
    """
    if method == "mean":
        col_mean = np.mean(matrix, axis=0)
        if np.any(np.abs(col_mean) < 1e-12):
            raise ValueError("均值化时某列均值为零，无法进行无量纲化。")
        return matrix / col_mean
    elif method == "init":
        col_init = matrix[0, :]
        if np.any(np.abs(col_init) < 1e-12):
            raise ValueError("初值化时某列第一个值为零，无法进行无量纲化。")
        return matrix / col_init
    else:
        raise ValueError(f"不支持的无量纲化方法：{method}，可选 'mean' 或 'init'。")


def grey_relational_coefficients(
    matrix: ndarray,
    reference: ndarray | None = None,
    rho: float = 0.5,
) -> ndarray:
    """计算灰色关联系数矩阵。

    对每个方案（行）的每个指标（列），计算其与参考序列对应值的关联系数。

    .. math::
        \\xi_{ij} = \\frac{\\min_i \\min_j \\Delta_{ij} + \\rho \\cdot
        \\max_i \\max_j \\Delta_{ij}}{\\Delta_{ij} + \\rho \\cdot
        \\max_i \\max_j \\Delta_{ij}}

    其中 :math:`\\Delta_{ij} = |reference_j - matrix_{ij}|`。

    Args:
        matrix: 已无量纲化的矩阵，形状为 ``(n, m)``。
        reference: 参考序列（理想方案），形状为 ``(m,)``。若为 None，则取每列最大值。
        rho: 分辨系数，取值范围 ``(0, 1]``，通常取 0.5。rho 越小，分辨能力越强。

    Returns:
        关联系数矩阵，形状为 ``(n, m)``，每个元素在 (0, 1] 之间。

    Raises:
        ValueError: rho 不在 (0, 1] 范围内，或者矩阵维数不对。
    """
    if matrix.ndim != 2:
        raise ValueError("matrix 必须是二维数组。")

    n, m = matrix.shape
    if n == 0 or m == 0:
        raise ValueError("matrix 不能为空。")

    if not (0.0 < rho <= 1.0):
        raise ValueError(f"分辨系数 rho 必须在 (0, 1] 范围内，得到 {rho}。")

    # 确定参考序列
    if reference is None:
        reference = np.max(matrix, axis=0)
    else:
        reference = np.asarray(reference, dtype=float)
        if reference.shape != (m,):
            raise ValueError(
                f"参考序列形状应为 ({m},)，实际为 {reference.shape}。"
            )

    # 计算绝对差矩阵
    diff = np.abs(matrix - reference)

    # 计算两极差
    min_diff = np.min(diff)
    max_diff = np.max(diff)

    if np.isclose(max_diff, 0.0):
        # 所有方案都与参考序列完全一致，关联系数全为 1
        return np.ones((n, m))

    # 计算关联系数
    coefficients = (min_diff + rho * max_diff) / (diff + rho * max_diff)

    return coefficients


def grey_relational_grades(
    coefficients: ndarray,
    weights: ndarray | None = None,
) -> ndarray:
    """根据关联系数矩阵计算各方案的灰色关联度。

    关联度为各方案关联系数的加权平均。

    .. math::
        r_i = \\sum_{j=1}^{m} w_j \\cdot \\xi_{ij}

    Args:
        coefficients: 关联系数矩阵，形状为 ``(n, m)``。
        weights: 指标权重向量，形状为 ``(m,)``。若为 None，则采用等权重。

    Returns:
        各方案的灰色关联度向量，形状为 ``(n,)``。关联度越大，方案越优。

    Raises:
        ValueError: 权重形状不匹配或权重含负值。
    """
    n, m = coefficients.shape

    if weights is None:
        weights = np.ones(m) / m
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (m,):
            raise ValueError(
                f"权重向量形状应为 ({m},)，实际为 {weights.shape}。"
            )
        if np.any(weights < 0):
            raise ValueError("权重不能为负数。")
        weights = weights / np.sum(weights)

    grades = np.sum(coefficients * weights, axis=1)
    return grades


def grey_relational_analysis(
    matrix: ndarray,
    kinds: list[int] | None = None,
    best_values: list[float | None] | None = None,
    intervals: list[tuple[float, float] | None] | None = None,
    reference: ndarray | None = None,
    weights: ndarray | None = None,
    rho: float = 0.5,
    normalize_method: str = "mean",
) -> dict:
    """灰色关联分析一站式接口。

    依次执行正向化 → 无量纲化 → 计算关联系数 → 计算关联度 → 排序。

    Args:
        matrix: 原始评价矩阵，形状为 ``(n, m)``，n 个方案，m 个指标。
        kinds: 可选，每列指标类型。若提供则先正向化。
            取值：1=极大型，2=极小型，3=中间型，4=区间型。
            默认为 None，表示所有指标已为极大型。
        best_values: 中间型列对应的最优值，其他列可为 None。
        intervals: 区间型列对应的 ``(下限, 上限)``，其他列可为 None。
        reference: 参考序列（理想方案），形状为 ``(m,)``。
            若为 None，则取正向化且无量纲化后各列最大值。
        weights: 指标权重向量，形状为 ``(m,)``。若为 None，则采用等权重。
        rho: 分辨系数，取值范围 ``(0, 1]``，通常取 0.5。
        normalize_method: 无量纲化方法，``"mean"``（均值化）或 ``"init"``（初值化）。

    Returns:
        dict，包含以下键：
            - ``"converted"``: 正向化后矩阵 ``(n, m)``
            - ``"normalized"``: 无量纲化后矩阵 ``(n, m)``
            - ``"reference"``: 实际使用的参考序列 ``(m,)``
            - ``"diff"``: 绝对差矩阵 ``(n, m)``
            - ``"coefficients"``: 关联系数矩阵 ``(n, m)``
            - ``"grades"``: 关联度向量 ``(n,)``
            - ``"ranks"``: 排名向量 ``(n,)``，秩 1 为最优

    Raises:
        ValueError: 输入参数不合法时抛出。
    """
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2:
        raise ValueError("matrix 必须是二维数组。")

    n, m = data.shape
    if n == 0 or m == 0:
        raise ValueError("matrix 不能为空。")

    # 步骤 1：指标正向化
    if kinds is not None:
        converted = positive_transform(data, kinds, best_values, intervals)
    else:
        converted = data.copy()

    # 步骤 2：无量纲化
    normalized = normalize_data(converted, method=normalize_method)

    # 步骤 3：确定参考序列
    if reference is not None:
        ref = np.asarray(reference, dtype=float)
        if ref.shape != (m,):
            raise ValueError(f"参考序列形状应为 ({m},)，实际为 {ref.shape}。")
        _ref = ref
    else:
        _ref = np.max(normalized, axis=0)

    # 步骤 4 & 5：计算关联系数矩阵（内部完成绝对差与两极差）
    coefficients = grey_relational_coefficients(normalized, reference=_ref, rho=rho)

    # 步骤 6：计算关联度
    grades = grey_relational_grades(coefficients, weights=weights)

    # 排序（关联度越大越优）
    ranks = np.argsort(-grades) + 1

    diff = np.abs(normalized - _ref)

    return {
        "converted": converted,
        "normalized": normalized,
        "reference": _ref,
        "diff": diff,
        "coefficients": coefficients,
        "grades": grades,
        "ranks": ranks,
    }


def _read_weights_interactive(m: int) -> ndarray:
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


def interactive_grey_relational_analysis():
    """控制台交互式灰色关联分析入口。"""
    print("=== 灰色关联分析 (GRA) ===")
    n = read_int("请输入参评数目：", min_value=1)
    m = read_int("请输入指标数目：", min_value=1)

    print("\n指标类型说明：1=极大型，2=极小型，3=中间型，4=区间型")
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

    print("\n选择无量纲化方法：1=均值化（推荐），2=初值化")
    method_choice = input(">>> 方法（直接回车默认为均值化）：").strip()
    if method_choice == "2":
        normalize_method = "init"
    else:
        normalize_method = "mean"
    print(f"使用{'初值化' if normalize_method == 'init' else '均值化'}方法。")

    rho_str = input("\n请输入分辨系数 rho（0~1，直接回车默认 0.5）：").strip()
    if rho_str == "":
        rho = 0.5
    else:
        try:
            rho = float(rho_str)
        except ValueError:
            raise ValueError(f"分辨系数必须是数字，得到 {rho_str}")
        if not (0.0 < rho <= 1.0):
            raise ValueError(f"分辨系数 rho 必须在 (0, 1] 范围内，得到 {rho}")

    weights = _read_weights_interactive(m)

    # 执行分析
    result = grey_relational_analysis(
        matrix=matrix,
        kinds=kinds,
        best_values=best_values,
        intervals=intervals,
        weights=weights,
        rho=rho,
        normalize_method=normalize_method,
    )

    print("\n步骤 1：指标正向化...")
    print_matrix(result["converted"], title="正向化后的矩阵：")

    print(f"\n步骤 2：无量纲化（{normalize_method}）...")
    print_matrix(result["normalized"], title="无量纲化后的矩阵：")

    print("\n步骤 3：确定参考序列（取各列最大值）...")
    print(f"参考序列：{np.round(result['reference'], 4).tolist()}")

    print("\n步骤 4：计算绝对差矩阵...")
    print_matrix(result["diff"], title="绝对差矩阵 Δ：")

    print(f"\n步骤 5：计算灰色关联系数矩阵（ρ = {rho}）...")
    print_matrix(result["coefficients"], title="关联系数矩阵 ξ：")

    print("\n步骤 6：计算灰色关联度并排序...")

    print("\n=== 评价结果 ===")
    print(f"{'方案':<6}{'关联度 r':<14}{'排名':<6}")
    for i in range(n):
        rank_i = int(np.where(result["ranks"] == i + 1)[0][0]) + 1
        print(f"{i + 1:<6}{result['grades'][i]:<14.4f}{rank_i:<6}")

    best = int(np.argmax(result["grades"]) + 1)
    print(f"\n最优方案为：方案 {best}")


if __name__ == "__main__":
    interactive_grey_relational_analysis()
