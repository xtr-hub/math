import numpy as np
from numpy import ndarray
from src.data import load_raw
from enum import Enum

RI = [0, 0.0001, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59]

class WeightVectorType(Enum):
    ARIAVE = "ari_ave"
    GEOAVE = "geo_ave"
    EIGVEC = "eig_vec"# 推荐，其他两种方法仅用于评估

def is_valid_judgment_matrix(matrix : ndarray):
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

def calculate_weight_vector(matrix : ndarray, n : float, weight_vector_type : WeightVectorType):
    """计算特征值与权重向量
    返回 (lambda_max, weight_vector) — lambda_max 为最大特征值，weight_vector 为归一化权重向量。
    """

    match weight_vector_type:
        case WeightVectorType.ARIAVE:
            # 算术平均法: 每列归一化后按行求平均, 再归一化
            col_sum = np.sum(matrix, axis=0)
            norm_matrix = matrix / col_sum  # 每列和为 1
            weight = np.sum(norm_matrix, axis=1)  # 按行求和
            weight = weight / np.sum(weight)  # 归一化

        case WeightVectorType.GEOAVE:
            # 几何平均法: 每行求几何平均, 再归一化
            row_prod = np.prod(matrix, axis=1)  # 每行的乘积
            geo_mean = row_prod ** (1 / n)  # 开 n 次方
            weight = geo_mean / np.sum(geo_mean)  # 归一化

        case WeightVectorType.EIGVEC:
            # 特征向量法: 最大特征值对应的特征向量, 归一化
            eig_vals, eig_vecs = np.linalg.eig(matrix)
            max_idx = np.argmax(np.real(eig_vals))  # 实部最大的特征值索引
            weight = np.real(eig_vecs[:, max_idx])  # 取对应特征向量的实部
            weight = weight / np.sum(weight)  # 归一化

    # 通过归一化后的权重向量反推 lambda_max: Aw = λ_max * w
    Aw = matrix @ weight
    # λ_max = 均值 (Aw_i / w_i)
    lambda_max = np.mean(Aw / weight)

    return lambda_max, weight
            
def calculate_weights(matrix : ndarray, weight_vector_type=WeightVectorType.EIGVEC):
    """计算判断矩阵的权重"""
    n = matrix.shape[0]
    if n <= 2:
        print("此矩阵n<=2，一定是一致性矩阵")
        return
    # 先校验是否是判断矩阵
    is_valid_judgment_matrix(matrix)
    # 获取最大特征值与权重向量，这里不需要权重向量
    lambda_max, _ = calculate_weight_vector(matrix, n,  weight_vector_type)
    print(lambda_max)
    # 利用公式计算CI
    CI = (lambda_max - n) / (n - 1)
    # 查表获取RI并计算CI/RI
    CR = CI / RI[n - 1]
    # 最后给出评价
    if CR < 0.10:
        print(f"CR为： {CR} ，一致性可以接受")
    else:
        print(f"CR为： {CR} ，一致性不可接受")