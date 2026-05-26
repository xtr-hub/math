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

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_idx = np.argmax(eigenvalues.real)
    lambda_max = eigenvalues[max_idx].real

    weights = eigenvectors[:, max_idx].real
    weights = weights / weights.sum()

    n = matrix.shape[0]
    RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    CI = (lambda_max - n) / (n - 1)
    CR = CI / RI[n]

    if CR >= 0.1:
        raise ValueError(f"一致性检验未通过: CR={CR:.4f} >= 0.1")

    return weights, lambda_max, CR


# 执行
df = load_raw("judgment_matrix.csv")
data_name = df.columns[1:].tolist()
matrix_values = df.iloc[:, 1:].values.astype(float)

is_valid_judgment_matrix(matrix_values)
weights, lambda_max, CR = calculate_weights(matrix_values)
print(f"权重: {weights}")
print(f"最大特征值: {lambda_max}")
print(f"CR: {CR}")