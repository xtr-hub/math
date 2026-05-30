import numpy as np
from numpy import ndarray
from src.data import load_raw
from enum import Enum

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

def calaulate_weights_vectors(matrix : ndarray, weight_vector_type : WeightVectorType):
    """归一化矩阵并返回权重向量"""
    match weight_vector_type:
        case WeightVectorType.EIGVEC:
            # 这里按列求平均
            n = matrix.shape[0]
            a_sum = np.sum(matrix, axis=0)
            stand_a = matrix / a_sum # 这里算完后每列的和为1,矩阵所有数之和为n
            wig_vec = np.sum(stand_a, axis=1) / n
            print(wig_vec)
            
def calculate_weights(matrix : ndarray, weight_vector_type=WeightVectorType.EIGVEC):
    """计算判断矩阵的权重"""
    # 先校验是否是判断矩阵
    is_valid_judgment_matrix(matrix)
    # 先归一化矩阵
    
    # 计算特征向量
    # 利用公式计算CI
    # 查表获取RI并计算CI/RI
    # 最后给出评价