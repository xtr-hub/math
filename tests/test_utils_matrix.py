import numpy as np
import pytest

import src.utils.matrix as um


class TestPositiveTransform:
    def test_benefit_unchanged(self):
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = um.positive_transform(matrix, [1, 1])
        assert np.allclose(result, matrix)

    def test_cost_to_benefit(self):
        matrix = np.array([[80.0, 90.0], [85.0, 85.0], [90.0, 80.0]])
        result = um.positive_transform(matrix, [1, 2])
        expected = np.array([
            [80.0, 0.0],
            [85.0, 5.0],
            [90.0, 10.0],
        ])
        assert np.allclose(result, expected)

    def test_intermediate(self):
        matrix = np.array([[1.0], [3.0], [5.0]])
        result = um.positive_transform(matrix, [3], best_values=[3.0])
        expected = np.array([[0.0], [1.0], [0.0]])
        assert np.allclose(result, expected)

    def test_intermediate_all_same(self):
        matrix = np.array([[3.0], [3.0], [3.0]])
        result = um.positive_transform(matrix, [3], best_values=[3.0])
        assert np.allclose(result, 1.0)

    def test_interval(self):
        matrix = np.array([[1.0], [3.0], [5.0], [7.0]])
        result = um.positive_transform(matrix, [4], intervals=[(3.0, 5.0)])
        expected = np.array([
            [0.0],
            [1.0],
            [1.0],
            [0.0],
        ])
        assert np.allclose(result, expected)

    def test_interval_all_inside(self):
        matrix = np.array([[3.0], [4.0], [5.0]])
        result = um.positive_transform(matrix, [4], intervals=[(3.0, 5.0)])
        assert np.allclose(result, 1.0)

    def test_invalid_kind(self):
        matrix = np.array([[1.0]])
        with pytest.raises(ValueError, match="不支持的指标类型"):
            um.positive_transform(matrix, [5])

    def test_missing_best_value(self):
        matrix = np.array([[1.0]])
        with pytest.raises(ValueError, match="中间型"):
            um.positive_transform(matrix, [3])

    def test_missing_interval(self):
        matrix = np.array([[1.0]])
        with pytest.raises(ValueError, match="区间型"):
            um.positive_transform(matrix, [4])

    def test_invalid_interval_order(self):
        matrix = np.array([[1.0]])
        with pytest.raises(ValueError, match="下限"):
            um.positive_transform(matrix, [4], intervals=[(5.0, 3.0)])


class TestVectorNormalize:
    def test_column_norms_are_one(self):
        matrix = np.array([
            [3.0, 4.0],
            [4.0, 3.0],
        ])
        normalized = um.vector_normalize(matrix)
        norms = np.linalg.norm(normalized, axis=0)
        assert np.allclose(norms, [1.0, 1.0])
        assert np.allclose(normalized[:, 0], [3 / 5, 4 / 5])
        assert np.allclose(normalized[:, 1], [4 / 5, 3 / 5])

    def test_zero_column_raises(self):
        matrix = np.array([
            [0.0, 1.0],
            [0.0, 2.0],
        ])
        with pytest.raises(ValueError, match="全零列"):
            um.vector_normalize(matrix)


class TestSumNormalize:
    def test_column_sums_are_one(self):
        matrix = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
        normalized = um.sum_normalize(matrix)
        assert np.allclose(np.sum(normalized, axis=0), [1.0, 1.0])
        assert np.allclose(normalized[:, 0], [1 / 4, 3 / 4])
        assert np.allclose(normalized[:, 1], [2 / 6, 4 / 6])

    def test_zero_column_raises(self):
        matrix = np.array([
            [0.0, 1.0],
            [0.0, 2.0],
        ])
        with pytest.raises(ValueError, match="列和为 0"):
            um.sum_normalize(matrix)
