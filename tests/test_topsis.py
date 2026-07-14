import numpy as np
import pytest

import src.algorithms.topsis as topsis

SAMPLE_MATRIX = np.array([
    [80.0, 90.0],
    [85.0, 85.0],
    [90.0, 80.0],
])


def test_convert_indicators_min_to_max():
    matrix = SAMPLE_MATRIX.copy()
    kinds = [1, 2]
    converted = topsis.convert_indicators(matrix, kinds)
    expected = np.array([
        [80.0, 0.0],
        [85.0, 5.0],
        [90.0, 10.0],
    ])
    assert np.allclose(converted, expected)


def test_normalize_matrix():
    matrix = np.array([
        [3.0, 4.0],
        [4.0, 3.0],
    ])
    normalized = topsis.normalize_matrix(matrix)
    norms = np.linalg.norm(normalized, axis=0)
    assert np.allclose(norms, [1.0, 1.0])
    assert np.allclose(normalized[:, 0], [3 / 5, 4 / 5])
    assert np.allclose(normalized[:, 1], [4 / 5, 3 / 5])


def test_weighted_normalized_matrix():
    normalized = np.array([
        [0.6, 0.8],
        [0.8, 0.6],
    ])
    weights = np.array([0.5, 0.5])
    weighted = topsis.weighted_normalized_matrix(normalized, weights)
    expected = np.array([
        [0.3, 0.4],
        [0.4, 0.3],
    ])
    assert np.allclose(weighted, expected)


def test_ideal_solutions():
    weighted = np.array([
        [0.1, 0.5],
        [0.3, 0.2],
        [0.2, 0.8],
    ])
    v_pos, v_neg = topsis.ideal_solutions(weighted)
    assert np.allclose(v_pos, [0.3, 0.8])
    assert np.allclose(v_neg, [0.1, 0.2])


def test_calculate_closeness():
    weighted = np.array([
        [0.1, 0.5],
        [0.3, 0.2],
        [0.2, 0.8],
    ])
    v_pos = np.array([0.3, 0.8])
    v_neg = np.array([0.1, 0.2])
    closeness = topsis.calculate_closeness(weighted, v_pos, v_neg)
    assert closeness.shape == (3,)
    # 方案 3 最接近正理想解，贴近度最大
    assert np.argmax(closeness) == 2


def test_normalize_matrix_zero_column_raises():
    matrix = np.array([
        [0.0, 1.0],
        [0.0, 2.0],
    ])
    with pytest.raises(ValueError, match="全零列"):
        topsis.normalize_matrix(matrix)
