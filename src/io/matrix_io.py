"""控制台通用的矩阵与数值输入输出工具。"""

import numpy as np
from numpy import ndarray


def read_int(prompt: str, min_value: int | None = None, max_value: int | None = None) -> int:
    """从控制台读取整数，支持可选的范围校验。"""
    while True:
        try:
            value = int(input(prompt).strip())
        except ValueError:
            print("请输入整数。")
            continue
        if min_value is not None and value < min_value:
            print(f"输入必须 ≥ {min_value}，请重新输入。")
            continue
        if max_value is not None and value > max_value:
            print(f"输入必须 ≤ {max_value}，请重新输入。")
            continue
        return value


def read_floats(
    prompt: str, count: int | None = None, positive: bool = False
) -> list[float]:
    """从控制台读取一行浮点数，支持数量和正数校验。"""
    while True:
        line = input(prompt).strip()
        try:
            values = [float(x) for x in line.split()]
        except ValueError:
            print("请输入数字，并用空格分隔。")
            continue
        if count is not None and len(values) != count:
            print(f"需要输入 {count} 个值，实际输入 {len(values)} 个，请重新输入。")
            continue
        if positive and any(v <= 0 for v in values):
            print("所有元素必须为正数，请重新输入。")
            continue
        return values


def read_reciprocal_matrix() -> ndarray:
    """交互式读取正互反判断矩阵，只输入上三角，下三角自动填充。"""
    print("\n=== 输入判断矩阵 ===")
    print("判断矩阵是 n×n 正互反矩阵：对角线为 1，a[j][i] = 1 / a[i][j]")
    print("标度规则：1 同等重要 | 3 稍微重要 | 5 明显重要 | 7 强烈重要 | 9 极端重要")
    print("          也可使用 2、4、6、8 等中间值。")

    n = read_int("\n请输入判断矩阵的阶数 n（≥2）：", min_value=2)

    matrix = np.ones((n, n), dtype=float)

    for i in range(n - 1):
        count = n - i - 1
        vals = read_floats(
            f"\n请输入第 {i + 1} 行右侧 {count} 个元素 "
            f"a[{i + 1}][{i + 2}] ~ a[{i + 1}][{n}]，用空格分隔：",
            count=count,
            positive=True,
        )
        for k, val in enumerate(vals):
            j = i + 1 + k
            matrix[i, j] = val
            matrix[j, i] = 1 / val

        print_matrix(matrix, title="当前矩阵：")

    return matrix

def read_matrix(n_rows: int, n_cols: int) -> ndarray:
    """读取普通 n_rows × n_cols 矩阵，按行输入。"""
    print(f"请输入 {n_rows} 行 {n_cols} 列矩阵，每行输入后按回车：")
    matrix = np.zeros((n_rows, n_cols), dtype=float)
    for i in range(n_rows):
        row_values = read_floats(
            f"请输入第 {i + 1} 行内容，用空格分隔：",
            count=n_cols,
        )
        matrix[i, :] = row_values
    print(f"已输入 {n_rows} 行 {n_cols} 列矩阵")
    return matrix

def print_matrix(matrix: ndarray, title: str | None = None, decimals: int = 4) -> None:
    """控制台打印矩阵。"""
    if title:
        print(title)
    print(np.round(matrix, decimals))
