"""模糊综合评价算法实现。

模糊综合评价（Fuzzy Comprehensive Evaluation）是一种处理评价信息具有
模糊性的多指标决策方法。基本流程：

1. 确定因素集 U = {u_1, u_2, ..., u_m}。
2. 确定评语集 V = {v_1, v_2, ..., v_p}。
3. 确定权重向量 A = (a_1, a_2, ..., a_m)。
4. 建立模糊综合评价矩阵 R（m×p），其中 r_ij 表示因素 u_i 对评语 v_j
   的隶属度。
5. 选择合成算子，计算综合评价向量 B = A ∘ R。
6. 根据 B 计算综合得分并判定等级（可选）。
"""

# ruff: noqa: E402
import sys
from enum import Enum
from pathlib import Path
from typing import cast

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
    read_floats,
    read_matrix,
)
from src.utils.matrix import positive_transform


class FuzzyOperator(Enum):
    """模糊综合评价合成算子。"""

    WEIGHTED_AVERAGE = "weighted_average"  # M(·,+)
    MAIN_FACTOR_PROMINENCE = "main_factor_prominence"  # M(∧,∨)
    MAIN_FACTOR_DECISION = "main_factor_decision"  # M(·,∨)
    BOUNDED_SUM = "bounded_sum"  # M(∧,⊕)


class MembershipMethod(Enum):
    """隶属度矩阵自动生成方法。"""

    TRAPEZOID = "trapezoid"  # 梯形隶属函数
    PIECEWISE = "piecewise"  # 分段硬划分


def _operator_name(operator: FuzzyOperator) -> str:
    """返回合成算子的中文名称。"""
    names = {
        FuzzyOperator.WEIGHTED_AVERAGE: "加权平均型 M(·,+)",
        FuzzyOperator.MAIN_FACTOR_PROMINENCE: "主因素突出型 M(∧,∨)",
        FuzzyOperator.MAIN_FACTOR_DECISION: "主因素决定型 M(·,∨)",
        FuzzyOperator.BOUNDED_SUM: "取小上界和型 M(∧,⊕)",
    }
    return names.get(operator, operator.value)


def _method_name(method: MembershipMethod) -> str:
    """返回隶属函数方法的中文名称。"""
    names = {
        MembershipMethod.TRAPEZOID: "梯形隶属函数",
        MembershipMethod.PIECEWISE: "分段硬划分",
    }
    return names.get(method, method.value)


def validate_weights(weights: ndarray, n_indicators: int | None = None) -> ndarray:
    """校验并归一化权重向量。

    Args:
        weights: 权重向量，形状为 (m,)。
        n_indicators: 可选，期望的指标数。

    Returns:
        归一化后的权重向量，形状为 (m,)，所有元素非负且和为 1。

    Raises:
        ValueError: 权重维度、符号或总和不合法。
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError("权重 weights 必须是一维数组")
    if weights.size == 0:
        raise ValueError("权重 weights 不能为空")
    if n_indicators is not None and len(weights) != n_indicators:
        raise ValueError(
            f"权重维度 {len(weights)} 与指标数 {n_indicators} 不一致"
        )
    if np.any(weights < 0):
        raise ValueError("权重不能为负数")
    total = np.sum(weights)
    if np.isclose(total, 0.0):
        raise ValueError("权重总和不能为 0")
    return weights / total


def validate_membership_matrix(
    R: ndarray, n_indicators: int | None = None
) -> ndarray:
    """校验隶属度矩阵。

    Args:
        R: 隶属度矩阵，形状为 (m, p)。
        n_indicators: 可选，期望的指标数。

    Returns:
        校验并裁剪到 [0, 1] 后的隶属度矩阵。

    Raises:
        ValueError: 矩阵形状或元素范围不合法。
    """
    R = np.asarray(R, dtype=float)
    if R.ndim != 2:
        raise ValueError("隶属度矩阵 R 必须是二维数组")
    if n_indicators is not None and R.shape[0] != n_indicators:
        raise ValueError(
            f"隶属度矩阵行数 {R.shape[0]} 与指标数 {n_indicators} 不一致"
        )
    if np.any(R < -1e-12) or np.any(R > 1.0 + 1e-12):
        raise ValueError("隶属度矩阵元素必须在 [0, 1] 之间")
    return np.clip(R, 0.0, 1.0)


def apply_operator(
    weights: ndarray, R: ndarray, operator: FuzzyOperator
) -> ndarray:
    """根据指定合成算子计算综合评价向量 B。

    Args:
        weights: 指标权重向量，形状为 (m,)。
        R: 隶属度矩阵，形状为 (m, p)。
        operator: 合成算子。

    Returns:
        综合评价向量 B，形状为 (p,)。

    Raises:
        ValueError: 维度不匹配或算子不支持。
    """
    weights = validate_weights(weights)
    R = validate_membership_matrix(R)
    if len(weights) != R.shape[0]:
        raise ValueError(
            f"权重维度 {len(weights)} 与隶属度矩阵行数 {R.shape[0]} 不一致"
        )

    if operator == FuzzyOperator.WEIGHTED_AVERAGE:
        # M(·,+): b_j = sum_i a_i * r_ij
        return weights @ R

    if operator == FuzzyOperator.MAIN_FACTOR_PROMINENCE:
        # M(∧,∨): b_j = max_i min(a_i, r_ij)
        return np.max(np.minimum(weights[:, None], R), axis=0)

    if operator == FuzzyOperator.MAIN_FACTOR_DECISION:
        # M(·,∨): b_j = max_i (a_i * r_ij)
        return np.max(weights[:, None] * R, axis=0)

    if operator == FuzzyOperator.BOUNDED_SUM:
        # M(∧,⊕): b_j = min(1, sum_i min(a_i, r_ij))
        return np.minimum(
            1.0, np.sum(np.minimum(weights[:, None], R), axis=0)
        )

    raise ValueError(f"不支持的合成算子：{operator}")


def normalize_b(b: ndarray) -> ndarray:
    """将综合评价向量 B 归一化，使其元素和为 1。

    Args:
        b: 综合评价向量，形状为 (p,)。

    Returns:
        归一化后的向量。
    """
    b = np.asarray(b, dtype=float)
    total = np.sum(b)
    if np.isclose(total, 0.0):
        return np.ones_like(b) / b.size
    return b / total


def fuzzy_comprehensive_evaluate(
    weights: ndarray,
    R: ndarray,
    operator: FuzzyOperator = FuzzyOperator.WEIGHTED_AVERAGE,
    scores: ndarray | list[float] | None = None,
    grades: list[str] | None = None,
) -> dict:
    """执行模糊综合评价。

    Args:
        weights: 指标权重向量，形状为 (m,)。
        R: 隶属度矩阵，形状为 (m, p)。
        operator: 合成算子，默认加权平均型。
        scores: 可选，评语集的量化分值，长度为 p。
        grades: 可选，评语等级名称列表，长度为 p。

    Returns:
        包含以下键的字典：
        - "operator": 使用的合成算子。
        - "B": 综合评价向量。
        - "B_normalized": 归一化后的综合评价向量。
        - "score": 若提供 scores，则为综合得分。
        - "grade_index": 若提供 scores，则为最大隶属度对应的等级索引。
        - "grade": 若同时提供 grades，则为判定的等级名称。

    Raises:
        ValueError: 输入参数不合法。
    """
    R_arr = validate_membership_matrix(R)
    weights_arr = validate_weights(weights, n_indicators=R_arr.shape[0])
    b = apply_operator(weights_arr, R_arr, operator)
    b_norm = normalize_b(b)

    result: dict = {
        "operator": operator,
        "B": b,
        "B_normalized": b_norm,
    }

    if scores is not None:
        scores_arr = np.asarray(scores, dtype=float)
        if len(scores_arr) != len(b):
            raise ValueError(
                f"scores 长度 {len(scores_arr)} 与评语等级数 {len(b)} 不一致"
            )
        result["score"] = float(b @ scores_arr)
        result["grade_index"] = int(np.argmax(b))
        if grades is not None:
            if len(grades) != len(b):
                raise ValueError(
                    f"grades 长度 {len(grades)} 与评语等级数 {len(b)} 不一致"
                )
            result["grade"] = grades[result["grade_index"]]

    return result


def _infer_boundaries(
    data: ndarray, boundaries: ndarray, n_grades: int
) -> ndarray:
    """根据用户输入的边界和数据范围推断完整边界。

    完整边界数量为 n_grades + 1。
    """
    boundaries = np.asarray(boundaries, dtype=float)
    n_bounds = len(boundaries)

    if n_bounds == n_grades + 1:
        full_bounds = boundaries
    elif n_bounds == n_grades - 1:
        data_min = float(np.min(data))
        data_max = float(np.max(data))
        full_bounds = np.concatenate(
            [[data_min], boundaries, [data_max]]
        )
    else:
        raise ValueError(
            f"等级数为 {n_grades} 时，边界数量应为 {n_grades + 1} "
            f"或 {n_grades - 1}（内部阈值），实际得到 {n_bounds}"
        )

    if not np.all(np.diff(full_bounds) > 0):
        raise ValueError("边界必须严格递增")
    return full_bounds


def _trapezoid_membership_vector(
    x: ndarray, boundaries: ndarray
) -> ndarray:
    """对每个 x 计算各等级的梯形隶属度。

    Args:
        x: 一维数组，形状为 (n,)。
        boundaries: 完整边界，形状为 (n_grades + 1,)。

    Returns:
        隶属度矩阵，形状为 (n, n_grades)。
    """
    x = np.asarray(x, dtype=float)
    boundaries = np.asarray(boundaries, dtype=float)
    n_grades = len(boundaries) - 1
    n = x.shape[0]
    R = np.zeros((n, n_grades))

    c = boundaries
    for k in range(n_grades):
        if k == 0:
            # 最左侧等级：x <= c[0] 时为 1，c[0] 到 c[1] 下降到 0
            left = np.ones(n)
            slope = np.where(
                c[1] > c[0], (c[1] - x) / (c[1] - c[0]), 0.0
            )
            R[:, k] = np.where(x <= c[0], left, np.where(x >= c[1], 0.0, slope))
        elif k == n_grades - 1:
            # 最右侧等级：c[n-1] 到 c[n] 上升到 1，x >= c[n] 时为 1
            slope = np.where(
                c[n_grades] > c[n_grades - 1],
                (x - c[n_grades - 1]) / (c[n_grades] - c[n_grades - 1]),
                0.0,
            )
            R[:, k] = np.where(
                x <= c[n_grades - 1], 0.0, np.where(x >= c[n_grades], 1.0, slope)
            )
        else:
            # 中间等级
            # 上升段 [c[k-1], c[k]]，平台 [c[k], c[k+1]]，下降段 [c[k+1], c[k+2]]
            rising = np.where(
                c[k] > c[k - 1], (x - c[k - 1]) / (c[k] - c[k - 1]), 0.0
            )
            falling = np.where(
                c[k + 2] > c[k + 1],
                (c[k + 2] - x) / (c[k + 2] - c[k + 1]),
                0.0,
            )
            R[:, k] = np.where(
                x <= c[k - 1],
                0.0,
                np.where(
                    x <= c[k],
                    rising,
                    np.where(x <= c[k + 1], 1.0, np.where(x <= c[k + 2], falling, 0.0)),
                ),
            )

    return R


def _piecewise_membership_vector(
    x: ndarray, boundaries: ndarray
) -> ndarray:
    """对每个 x 计算各等级的分段硬划分隶属度。

    Args:
        x: 一维数组，形状为 (n,)。
        boundaries: 完整边界，形状为 (n_grades + 1,)。

    Returns:
        隶属度矩阵，形状为 (n, n_grades)。
    """
    x = np.asarray(x, dtype=float)
    boundaries = np.asarray(boundaries, dtype=float)
    n_grades = len(boundaries) - 1
    n = x.shape[0]
    R = np.zeros((n, n_grades))

    for k in range(n_grades):
        lower = boundaries[k]
        upper = boundaries[k + 1]
        if k == n_grades - 1:
            mask = (x >= lower) & (x <= upper)
        else:
            mask = (x >= lower) & (x < upper)
        R[:, k] = mask.astype(float)

    return R


def build_membership_matrix(
    raw_matrix: ndarray,
    boundaries: ndarray | list[float],
    n_grades: int | None = None,
    method: MembershipMethod = MembershipMethod.TRAPEZOID,
    kinds: list[int] | None = None,
    best_values: list[float | None] | None = None,
    intervals: list[tuple[float, float] | None] | None = None,
) -> ndarray:
    """根据原始指标矩阵和评语等级边界自动生成隶属度矩阵。

    返回的三维数组形状为 (n, m, p)，其中 n 为方案数，m 为指标数，
    p 为评语等级数。R[i] 即为第 i 个方案的模糊评价矩阵。

    Args:
        raw_matrix: 原始评价矩阵，形状为 (n, m)。
        boundaries: 等级边界。若提供 n_grades + 1 个值，则视为完整边界；
            若提供 n_grades - 1 个值，则视为内部阈值，两端自动使用数据最值。
        n_grades: 评语等级数。若未提供，则根据 boundaries 长度推断
            （假设为完整边界）。
        method: 隶属函数方法，梯形或分段硬划分。
        kinds: 可选，每列指标类型（1=极大型，2=极小型，3=中间型，4=区间型）。
            若提供，则先进行正向化。
        best_values: 中间型列对应的最优值。
        intervals: 区间型列对应的满意区间。

    Returns:
        隶属度张量，形状为 (n, m, p)。

    Raises:
        ValueError: 参数不合法。
    """
    data = np.asarray(raw_matrix, dtype=float)
    if data.ndim != 2:
        raise ValueError("raw_matrix 必须是二维数组")

    if kinds is not None:
        data = positive_transform(data, kinds, best_values, intervals)

    boundaries_arr = np.asarray(boundaries, dtype=float)
    if n_grades is None:
        n_grades = len(boundaries_arr) - 1
        if n_grades < 1:
            raise ValueError("无法从 boundaries 推断等级数")

    full_bounds = _infer_boundaries(data, boundaries_arr, n_grades)
    n_grades = len(full_bounds) - 1

    n, m = data.shape
    R = np.zeros((n, m, n_grades))

    if method == MembershipMethod.TRAPEZOID:
        for j in range(m):
            R[:, j, :] = _trapezoid_membership_vector(data[:, j], full_bounds)
    elif method == MembershipMethod.PIECEWISE:
        for j in range(m):
            R[:, j, :] = _piecewise_membership_vector(data[:, j], full_bounds)
    else:
        raise ValueError(f"不支持的隶属函数方法：{method}")

    return R


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


def _read_grade_names(n_grades: int) -> list[str]:
    """交互式读取评语等级名称，直接回车则使用默认名称。"""
    print(
        f"\n请输入 {n_grades} 个评语等级名称（用空格分隔），"
        "直接回车则使用默认名称："
    )
    while True:
        line = input(">>> 等级名称：").strip()
        if line == "":
            return [f"等级 {i + 1}" for i in range(n_grades)]
        names = line.split()
        if len(names) != n_grades:
            print(f"需要输入 {n_grades} 个名称，实际输入 {len(names)} 个。")
            continue
        return names


def prompt_for_operator() -> FuzzyOperator:
    """交互式选择合成算子。"""
    print("\n请选择合成算子：")
    print("1. 加权平均型 M(·,+)")
    print("2. 主因素突出型 M(∧,∨)")
    print("3. 主因素决定型 M(·,∨)")
    print("4. 取小上界和型 M(∧,⊕)")

    choice = read_int(">>> 请选择（1-4）：", min_value=1, max_value=4)
    mapping = {
        1: FuzzyOperator.WEIGHTED_AVERAGE,
        2: FuzzyOperator.MAIN_FACTOR_PROMINENCE,
        3: FuzzyOperator.MAIN_FACTOR_DECISION,
        4: FuzzyOperator.BOUNDED_SUM,
    }
    return mapping[choice]


def prompt_for_weight_source() -> str:
    """交互式选择权重来源。"""
    print("\n请选择权重来源：")
    print("1. 手动输入")
    print("2. 等权重")
    print("3. 熵权法")
    print("4. AHP")

    choice = read_int(">>> 请选择（1-4）：", min_value=1, max_value=4)
    mapping = {
        1: "manual",
        2: "equal",
        3: "entropy",
        4: "ahp",
    }
    return mapping[choice]


def prompt_for_membership_source() -> str:
    """交互式选择隶属度矩阵来源。"""
    print("\n请选择隶属度矩阵来源：")
    print("1. 手动输入")
    print("2. 根据原始指标矩阵自动生成")

    choice = read_int(">>> 请选择（1-2）：", min_value=1, max_value=2)
    return "manual" if choice == 1 else "auto"


def prompt_for_membership_method() -> MembershipMethod:
    """交互式选择隶属函数方法。"""
    print("\n请选择隶属函数方法：")
    print("1. 梯形隶属函数")
    print("2. 分段硬划分")

    choice = read_int(">>> 请选择（1-2）：", min_value=1, max_value=2)
    return (
        MembershipMethod.TRAPEZOID
        if choice == 1
        else MembershipMethod.PIECEWISE
    )


def _read_indicator_kinds(m: int) -> list[int]:
    """交互式读取指标类型。"""
    print("\n指标类型说明：1=极大型，2=极小型，3=中间型，4=区间型")
    kinds = read_ints(
        "请输入类型矩阵（用空格分隔）：",
        count=m,
        positive=False,
    )
    for k in kinds:
        if k not in (1, 2, 3, 4):
            raise ValueError(f"指标类型必须是 1/2/3/4 之一，得到 {k}")
    return kinds


def _compute_entropy_weights(
    matrix: ndarray, kinds: list[int], best_values, intervals
) -> ndarray:
    """函数内部导入熵权法模块，避免循环依赖。"""
    from src.algorithms.entropy_weight import calculate_entropy_weights

    return calculate_entropy_weights(
        matrix,
        kinds=kinds,
        best_values=best_values,
        intervals=intervals,
    )


def _compute_ahp_weights() -> ndarray:
    """函数内部导入 AHP 模块，避免循环依赖。"""
    from src.algorithms.ahp import WeightVectorType, calculate_weight_vector

    print("\n=== AHP 判断矩阵输入 ===")
    n = read_int("请输入判断矩阵阶数 n（≥2）：", min_value=2)
    matrix = np.ones((n, n), dtype=float)
    for i in range(n - 1):
        count = n - i - 1
        vals = read_floats(
            f"请输入第 {i + 1} 行右侧 {count} 个元素，用空格分隔：",
            count=count,
            positive=True,
        )
        for k, val in enumerate(vals):
            j = i + 1 + k
            matrix[i, j] = val
            matrix[j, i] = 1.0 / val
        print_matrix(matrix, title="当前判断矩阵：")

    print("\n请选择权重向量计算方法：")
    print("1. 特征向量法（推荐）")
    print("2. 算术平均法")
    print("3. 几何平均法")
    choice = read_int(">>> 请选择（1-3）：", min_value=1, max_value=3)
    type_mapping = {
        1: WeightVectorType.EIGVEC,
        2: WeightVectorType.ARIAVE,
        3: WeightVectorType.GEOAVE,
    }
    _, weights = calculate_weight_vector(
        matrix, n, type_mapping[choice]
    )
    return weights


def interactive_fuzzy_comprehensive_evaluation():
    """控制台交互式模糊综合评价入口。"""
    print("=== 模糊综合评价法 ===")

    n = read_int("请输入参评方案数：", min_value=1)
    m = read_int("请输入指标数：", min_value=1)
    n_grades = read_int("请输入评语等级数：", min_value=2)

    grades = _read_grade_names(n_grades)
    print(f"评语等级：{grades}")

    # 权重来源
    weight_source = prompt_for_weight_source()

    # 先处理不需要原始矩阵的权重来源
    weights: ndarray | None = None
    if weight_source == "manual":
        weights = read_weights(m)
    elif weight_source == "equal":
        weights = np.ones(m) / m
        print(f"使用等权重：{np.round(weights, 4).tolist()}")
    elif weight_source == "ahp":
        weights = _compute_ahp_weights()
        if len(weights) != m:
            raise ValueError("AHP 权重维度与指标数不一致")
        print(f"AHP 权重：{np.round(weights, 4).tolist()}")

    # 隶属度矩阵来源
    membership_source = prompt_for_membership_source()

    # 如果权重来源或 R 来源需要原始指标矩阵，则统一先输入
    need_raw_matrix = (
        weight_source == "entropy" or membership_source == "auto"
    )

    matrix: ndarray | None = None
    kinds: list[int] | None = None
    best_values: list[float | None] | None = None
    intervals: list[tuple[float, float] | None] | None = None

    if need_raw_matrix:
        print("\n指标类型说明：1=极大型，2=极小型，3=中间型，4=区间型")
        kinds = _read_indicator_kinds(m)
        best_values, intervals = read_indicator_params(kinds)
        print("\n请输入原始指标矩阵：")
        matrix = read_matrix(n, m)

    # 如果权重来源是熵权法，此时计算权重
    if weight_source == "entropy":
        if matrix is None or kinds is None:
            raise RuntimeError("熵权法需要原始指标矩阵")
        weights = _compute_entropy_weights(
            cast(ndarray, matrix), kinds, best_values, intervals
        )
        print(f"熵权法权重：{np.round(weights, 4).tolist()}")

    if weights is None:
        raise RuntimeError("权重未正确计算")

    # 构建隶属度矩阵
    if membership_source == "manual":
        R_list = []
        for i in range(n):
            print(f"\n请输入第 {i + 1} 个方案的隶属度矩阵（{m} 行 {n_grades} 列）：")
            R_i = read_matrix(m, n_grades)
            R_list.append(validate_membership_matrix(R_i, n_indicators=m))
        R_tensor = np.stack(R_list, axis=0)
    elif membership_source == "auto":
        if matrix is None or kinds is None:
            raise RuntimeError("自动生成隶属度矩阵需要原始指标矩阵")
        print(
            f"\n请输入 {n_grades - 1} 个内部阈值（用空格分隔），"
            "或直接输入 {n_grades + 1} 个完整边界："
        )
        boundaries = np.array(read_floats(">>> 边界："))
        method = prompt_for_membership_method()
        R_tensor = build_membership_matrix(
            cast(ndarray, matrix),
            boundaries,
            n_grades=n_grades,
            method=method,
            kinds=kinds,
            best_values=best_values,
            intervals=intervals,
        )
        print(f"\n使用 {_method_name(method)} 自动生成隶属度矩阵。")
    else:
        raise ValueError(f"不支持的隶属度矩阵来源：{membership_source}")

    # 选择算子
    operator = prompt_for_operator()

    # 是否输入量化分值
    print("\n请输入各等级的量化分值（用空格分隔），直接回车则不计算综合得分：")
    scores: ndarray | None = None
    while True:
        line = input(">>> 分值：").strip()
        if line == "":
            break
        try:
            scores = np.array([float(x) for x in line.split()])
        except ValueError:
            print("请输入数字，并用空格分隔。")
            continue
        if len(scores) != n_grades:
            print(f"需要输入 {n_grades} 个分值，实际输入 {len(scores)} 个。")
            continue
        break

    # 对每个方案进行评价
    print(f"\n使用 {_operator_name(operator)} 进行模糊合成...")
    print("\n=== 评价结果 ===")
    for i in range(n):
        R_i = R_tensor[i]
        result = fuzzy_comprehensive_evaluate(
            weights, R_i, operator, scores=scores, grades=grades
        )
        print(f"\n方案 {i + 1}：")
        print(f"  综合评价向量 B：{np.round(result['B'], 4).tolist()}")
        print(
            f"  归一化向量 B*：{np.round(result['B_normalized'], 4).tolist()}"
        )
        if "score" in result:
            print(f"  综合得分：{result['score']:.4f}")
        if "grade" in result:
            print(f"  判定等级：{result['grade']}")


if __name__ == "__main__":
    interactive_fuzzy_comprehensive_evaluation()
