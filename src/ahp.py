import numpy as np
from src.data import load_raw


def is_valid_judgment_matrix(matrix):
    """判断是否是有效的判断矩阵"""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("必须是方阵")
    n = matrix.shape[0]
    if n < 2:
        raise ValueError("判断矩阵至少为2x2矩阵")

    # 对角线必须为 1
    if not np.allclose(np.diag(matrix), 1):
        raise ValueError("对角线元素必须为 1")

    for i in range(n):
        for j in range(i + 1, n):
            if not np.isclose(matrix[j, i], 1 / matrix[i, j]):
                raise ValueError(f"不满足互反性: a[{i},{j}]={matrix[i, j]}, a[{j},{i}]={matrix[j, i]}")

    if np.any(matrix <= 0):
        raise ValueError("所有元素必须为正数")
    return True


def calculate_weights(matrix):
    """计算判断矩阵的权重"""
    is_valid_judgment_matrix(matrix)
    