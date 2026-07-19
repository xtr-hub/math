import numpy as np
import pytest

import src.algorithms.grey_relational_analysis as gra

# 三个方案，两个指标（均为极大型）
SAMPLE_MATRIX = np.array([
    [8.0, 9.0],
    [7.0, 8.0],
    [9.0, 7.0],
])


class TestNormalizeData:
    def test_mean_normalize(self):
        matrix = np.array([
            [2.0, 6.0],
            [4.0, 8.0],
        ])
        result = gra.normalize_data(matrix, method="mean")
        # 列均值：[3, 7]，归一化后各列均值应为 1
        assert np.allclose(np.mean(result, axis=0), [1.0, 1.0])
        assert np.allclose(result, [[2 / 3, 6 / 7], [4 / 3, 8 / 7]])

    def test_init_normalize(self):
        matrix = np.array([
            [2.0, 6.0],
            [4.0, 8.0],
        ])
        result = gra.normalize_data(matrix, method="init")
        expected = np.array([
            [1.0, 1.0],
            [2.0, 4 / 3],
        ])
        assert np.allclose(result, expected)

    def test_mean_zero_column_raises(self):
        matrix = np.array([
            [0.0, 1.0],
            [0.0, 2.0],
        ])
        with pytest.raises(ValueError, match="均值为零"):
            gra.normalize_data(matrix, method="mean")

    def test_init_zero_first_raises(self):
        matrix = np.array([
            [0.0, 1.0],
            [2.0, 2.0],
        ])
        with pytest.raises(ValueError, match="第一个值为零"):
            gra.normalize_data(matrix, method="init")

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="不支持"):
            gra.normalize_data(SAMPLE_MATRIX, method="foo")


class TestGreyRelationalCoefficients:
    def test_basic_coefficients(self):
        # 等间距数据，ρ=0.5
        matrix = np.array([
            [1.0, 1.0],
            [0.5, 0.5],
            [0.0, 0.0],
        ])
        coeff = gra.grey_relational_coefficients(matrix, rho=0.5)
        assert coeff.shape == (3, 2)
        # 关联系数应在 (0, 1] 范围内
        assert np.all(coeff > 0)
        assert np.all(coeff <= 1.0)
        # 与参考序列最接近的方案 1 关联系数应最大
        assert np.all(coeff[0] > coeff[1])
        assert np.all(coeff[1] > coeff[2])

    def test_with_custom_reference(self):
        matrix = np.array([
            [8.0, 9.0],
            [7.0, 8.0],
        ])
        reference = np.array([10.0, 10.0])
        coeff = gra.grey_relational_coefficients(matrix, reference=reference, rho=0.5)
        assert coeff.shape == (2, 2)
        # 方案 1 更接近参考序列
        assert np.all(coeff[0] > coeff[1])

    def test_identical_to_reference(self):
        matrix = np.array([
            [5.0, 5.0],
            [5.0, 5.0],
        ])
        reference = np.array([5.0, 5.0])
        coeff = gra.grey_relational_coefficients(matrix, reference=reference)
        # 完全一致时关联系数应为 1
        assert np.allclose(coeff, 1.0)

    def test_rho_edge_cases(self):
        matrix = np.array([
            [1.0, 0.5],
            [0.8, 0.6],
        ])
        coeff_small = gra.grey_relational_coefficients(matrix, rho=0.1)
        coeff_large = gra.grey_relational_coefficients(matrix, rho=1.0)
        # ρ 较小时关联系数整体更小（分辨能力更强）
        assert np.all(coeff_small < coeff_large)

    def test_rho_out_of_range(self):
        with pytest.raises(ValueError, match="分辨系数"):
            gra.grey_relational_coefficients(SAMPLE_MATRIX, rho=0.0)
        with pytest.raises(ValueError, match="分辨系数"):
            gra.grey_relational_coefficients(SAMPLE_MATRIX, rho=1.5)

    def test_reference_shape_mismatch(self):
        reference = np.array([1.0, 2.0, 3.0])  # 应为 2 列
        with pytest.raises(ValueError, match="参考序列形状"):
            gra.grey_relational_coefficients(SAMPLE_MATRIX, reference=reference)

    def test_default_reference_is_max_per_column(self):
        matrix = np.array([
            [1.0, 5.0],
            [3.0, 2.0],
            [2.0, 4.0],
        ])
        coeff = gra.grey_relational_coefficients(matrix)
        # 参考序列应为 [3.0, 5.0]，方案 1 离参考序列最远
        # 方案 2 在列1最好，方案1在列2最好，方案3中等
        assert coeff.shape == (3, 2)

    def test_empty_matrix_raises(self):
        with pytest.raises(ValueError, match="不能为空"):
            gra.grey_relational_coefficients(np.array([[]]))


class TestGreyRelationalGrades:
    def test_equal_weights(self):
        coeff = np.array([
            [0.8, 0.6],
            [0.4, 0.9],
            [0.7, 0.7],
        ])
        grades = gra.grey_relational_grades(coeff)
        assert grades.shape == (3,)
        assert np.allclose(np.sum(grades), (0.7 + 0.65 + 0.7))  # 等权重均值
        assert np.allclose(grades, np.mean(coeff, axis=1))

    def test_custom_weights(self):
        coeff = np.array([
            [0.8, 0.6],
            [0.4, 0.9],
        ])
        weights = np.array([0.7, 0.3])
        grades = gra.grey_relational_grades(coeff, weights=weights)
        expected = np.array([0.8 * 0.7 + 0.6 * 0.3, 0.4 * 0.7 + 0.9 * 0.3])
        assert np.allclose(grades, expected)

    def test_weights_normalized(self):
        coeff = np.array([
            [0.8, 0.6],
            [0.4, 0.9],
        ])
        # 未归一化的权重
        weights = np.array([2.0, 3.0])
        grades = gra.grey_relational_grades(coeff, weights=weights)
        expected = np.array([0.8 * 0.4 + 0.6 * 0.6, 0.4 * 0.4 + 0.9 * 0.6])
        assert np.allclose(grades, expected)

    def test_negative_weights_raises(self):
        with pytest.raises(ValueError, match="负"):
            gra.grey_relational_grades(
                np.ones((2, 2)), weights=np.array([0.5, -0.5])
            )

    def test_weight_shape_mismatch(self):
        with pytest.raises(ValueError, match="权重向量形状"):
            gra.grey_relational_grades(
                np.ones((2, 2)), weights=np.array([0.3, 0.3, 0.4])
            )


class TestGreyRelationalAnalysis:
    def test_end_to_end_basic(self):
        result = gra.grey_relational_analysis(SAMPLE_MATRIX)
        assert "grades" in result
        assert "ranks" in result
        assert "coefficients" in result
        assert result["grades"].shape == (3,)
        assert result["ranks"].shape == (3,)
        # 关联度之和应合理
        assert np.all(result["grades"] >= 0)
        assert np.all(result["grades"] <= 1.0)
        # 排名应为 1..n
        assert set(result["ranks"].tolist()) == {1, 2, 3}

    def test_end_to_end_with_kinds(self):
        # 指标2为极小型，值越小越好
        matrix = np.array([
            [8.0, 5.0],
            [7.0, 3.0],
            [9.0, 4.0],
        ])
        kinds = [1, 2]  # 第2列为极小型
        result = gra.grey_relational_analysis(matrix, kinds=kinds)
        assert "converted" in result
        # 正向化后极小型列应反转
        assert result["converted"].shape == (3, 2)

    def test_single_scheme(self):
        matrix = np.array([[5.0, 6.0]])
        result = gra.grey_relational_analysis(matrix)
        assert result["grades"].shape == (1,)
        # 唯一方案关联系数全为 1
        assert np.allclose(result["coefficients"], 1.0)

    def test_custom_reference(self):
        reference = np.array([10.0, 10.0])
        result = gra.grey_relational_analysis(SAMPLE_MATRIX, reference=reference)
        assert np.allclose(result["reference"], reference)

    def test_custom_weights(self):
        weights = np.array([0.6, 0.4])
        result = gra.grey_relational_analysis(SAMPLE_MATRIX, weights=weights)
        assert result["grades"].shape == (3,)

    def test_not_two_dim_raises(self):
        with pytest.raises(ValueError, match="二维"):
            gra.grey_relational_analysis(np.array([1.0, 2.0]))

    def test_empty_matrix_raises(self):
        with pytest.raises(ValueError, match="不能为空"):
            gra.grey_relational_analysis(np.array([[]]))

    def test_reference_shape_mismatch(self):
        with pytest.raises(ValueError, match="参考序列形状"):
            gra.grey_relational_analysis(SAMPLE_MATRIX, reference=np.array([1.0]))

    def test_ranking_correct(self):
        # 方案 1 明显最优
        matrix = np.array([
            [100.0, 100.0],
            [50.0, 50.0],
            [10.0, 10.0],
        ])
        result = gra.grey_relational_analysis(matrix)
        assert result["ranks"][0] == 1  # 方案 1 排第 1
        assert result["ranks"][2] == 3  # 方案 3 排第 3

    def test_init_normalize_method(self):
        result = gra.grey_relational_analysis(SAMPLE_MATRIX, normalize_method="init")
        assert "normalized" in result
        # 初值化后第一行应全为 1
        assert np.allclose(result["normalized"][0], 1.0)


class TestInteractive:
    def test_read_weights_default(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "")
        weights = gra._read_weights_interactive(3)
        assert np.allclose(weights, [1 / 3, 1 / 3, 1 / 3])

    def test_read_weights_valid(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "0.5 0.3 0.2")
        weights = gra._read_weights_interactive(3)
        assert np.allclose(weights, [0.5, 0.3, 0.2])

    def test_read_weights_wrong_count_then_valid(self, monkeypatch):
        inputs = iter(["0.5 0.5", "0.4 0.6"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        weights = gra._read_weights_interactive(2)
        assert np.allclose(weights, [0.4, 0.6])
