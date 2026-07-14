import numpy as np
import pytest

import src.algorithms.entropy_weight as ew

SAMPLE_MATRIX = np.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [3.0, 3.0],
])


def test_entropy_weights_shape_and_sum():
    weights = ew.calculate_entropy_weights(SAMPLE_MATRIX)
    assert weights.shape == (2,)
    assert np.allclose(np.sum(weights), 1.0)
    assert np.all(weights >= 0)


def test_single_sample_returns_equal_weights():
    matrix = np.array([[5.0, 10.0]])
    weights = ew.calculate_entropy_weights(matrix)
    assert np.allclose(weights, [0.5, 0.5])


def test_uniform_column_gives_equal_weights():
    # 两列完全成比例（信息量相同），权重应相等
    matrix = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
    ])
    weights = ew.calculate_entropy_weights(matrix)
    assert np.allclose(weights, [0.5, 0.5])


def test_with_cost_indicator():
    # 第一列为成本型，正向化后列数据为 [2,1,0]；第二列极大型 [2,1,0]
    matrix = np.array([
        [2.0, 2.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ])
    weights = ew.calculate_entropy_weights(matrix, kinds=[2, 1])
    # 正向化后两列相同，熵相同，权重相等
    assert np.allclose(weights, [0.5, 0.5])


def test_intermediate_indicator():
    matrix = np.array([[1.0], [3.0], [5.0]])
    weights = ew.calculate_entropy_weights(
        matrix,
        kinds=[3],
        best_values=[3.0],
    )
    assert weights.shape == (1,)
    assert np.allclose(np.sum(weights), 1.0)


def test_interval_indicator():
    matrix = np.array([[1.0], [3.0], [5.0], [7.0]])
    weights = ew.calculate_entropy_weights(
        matrix,
        kinds=[4],
        intervals=[(3.0, 5.0)],
    )
    assert weights.shape == (1,)
    assert np.allclose(np.sum(weights), 1.0)


def test_negative_value_raises():
    matrix = np.array([
        [1.0, -1.0],
        [2.0, 3.0],
    ])
    with pytest.raises(ValueError, match="非负"):
        ew.calculate_entropy_weights(matrix)


def test_zero_column_returns_zero_weight():
    matrix = np.array([
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
    ])
    weights = ew.calculate_entropy_weights(matrix)
    # 第二列无差异，冗余度为 0，对应权重为 0
    assert np.allclose(weights, [1.0, 0.0])
