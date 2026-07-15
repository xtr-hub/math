import numpy as np
import pytest

import src.algorithms.fuzzy_comprehensive_evaluation as fuzzy


class TestValidateWeights:
    def test_validate_weights_normalizes(self):
        weights = np.array([1.0, 2.0, 3.0])
        normalized = fuzzy.validate_weights(weights)
        assert np.allclose(normalized, [1 / 6, 2 / 6, 3 / 6])

    def test_validate_weights_with_expected_count(self):
        weights = np.array([1.0, 1.0])
        normalized = fuzzy.validate_weights(weights, n_indicators=2)
        assert np.allclose(normalized, [0.5, 0.5])

    def test_validate_weights_rejects_negative(self):
        with pytest.raises(ValueError, match="权重不能为负数"):
            fuzzy.validate_weights(np.array([1.0, -1.0]))

    def test_validate_weights_rejects_zero_sum(self):
        with pytest.raises(ValueError, match="权重总和不能为 0"):
            fuzzy.validate_weights(np.array([0.0, 0.0]))

    def test_validate_weights_rejects_dimension_mismatch(self):
        with pytest.raises(ValueError, match="权重维度 2 与指标数 3 不一致"):
            fuzzy.validate_weights(np.array([0.5, 0.5]), n_indicators=3)

    def test_validate_weights_rejects_empty(self):
        with pytest.raises(ValueError, match="权重 weights 不能为空"):
            fuzzy.validate_weights(np.array([]))

    def test_validate_weights_rejects_non_1d(self):
        with pytest.raises(ValueError, match="权重 weights 必须是一维数组"):
            fuzzy.validate_weights(np.array([[0.5, 0.5]]))


class TestValidateMembershipMatrix:
    def test_validate_membership_matrix_pass(self):
        R = np.array([[0.2, 0.8], [0.5, 0.5]])
        validated = fuzzy.validate_membership_matrix(R)
        assert np.allclose(validated, R)

    def test_validate_membership_matrix_clips_tiny_errors(self):
        R = np.array([[-1e-13, 1.0 + 1e-13]])
        validated = fuzzy.validate_membership_matrix(R)
        assert np.allclose(validated, [0.0, 1.0])

    def test_validate_membership_matrix_rejects_negative(self):
        with pytest.raises(ValueError, match="隶属度矩阵元素必须在"):
            fuzzy.validate_membership_matrix(np.array([[-0.1, 0.5]]))

    def test_validate_membership_matrix_rejects_greater_than_one(self):
        with pytest.raises(ValueError, match="隶属度矩阵元素必须在"):
            fuzzy.validate_membership_matrix(np.array([[1.1, 0.5]]))

    def test_validate_membership_matrix_rejects_dimension_mismatch(self):
        R = np.array([[0.5, 0.5]])
        with pytest.raises(ValueError, match="隶属度矩阵行数 1 与指标数 2 不一致"):
            fuzzy.validate_membership_matrix(R, n_indicators=2)

    def test_validate_membership_matrix_rejects_non_2d(self):
        with pytest.raises(ValueError, match="隶属度矩阵 R 必须是二维数组"):
            fuzzy.validate_membership_matrix(np.array([0.5, 0.5]))


class TestApplyOperator:
    def test_weighted_average(self):
        weights = np.array([0.5, 0.5])
        R = np.array([[0.2, 0.8], [0.6, 0.4]])
        B = fuzzy.apply_operator(weights, R, fuzzy.FuzzyOperator.WEIGHTED_AVERAGE)
        assert np.allclose(B, [0.4, 0.6])

    def test_main_factor_prominence(self):
        weights = np.array([0.4, 0.6])
        R = np.array([[0.5, 0.5], [0.8, 0.2]])
        B = fuzzy.apply_operator(
            weights, R, fuzzy.FuzzyOperator.MAIN_FACTOR_PROMINENCE
        )
        # min(0.4, 0.5)=0.4, min(0.4, 0.5)=0.4; min(0.6, 0.8)=0.6, min(0.6, 0.2)=0.2
        # max over rows -> [0.6, 0.4]
        assert np.allclose(B, [0.6, 0.4])

    def test_main_factor_decision(self):
        weights = np.array([0.4, 0.6])
        R = np.array([[0.5, 0.5], [0.8, 0.2]])
        B = fuzzy.apply_operator(
            weights, R, fuzzy.FuzzyOperator.MAIN_FACTOR_DECISION
        )
        # 0.4*0.5=0.2, 0.4*0.5=0.2; 0.6*0.8=0.48, 0.6*0.2=0.12
        # max -> [0.48, 0.2]
        assert np.allclose(B, [0.48, 0.2])

    def test_bounded_sum(self):
        weights = np.array([0.5, 0.5])
        R = np.array([[0.8, 0.2], [0.9, 0.1]])
        B = fuzzy.apply_operator(weights, R, fuzzy.FuzzyOperator.BOUNDED_SUM)
        # min(0.5,0.8)+min(0.5,0.9)=0.5+0.5=1.0 -> min(1,1.0)=1.0
        # min(0.5,0.2)+min(0.5,0.1)=0.2+0.1=0.3
        assert np.allclose(B, [1.0, 0.3])

    def test_apply_operator_dimension_mismatch(self):
        weights = np.array([0.5, 0.5])
        R = np.array([[0.5, 0.5]])
        with pytest.raises(ValueError, match="权重维度 2 与隶属度矩阵行数 1 不一致"):
            fuzzy.apply_operator(weights, R, fuzzy.FuzzyOperator.WEIGHTED_AVERAGE)


class TestNormalizeB:
    def test_normalize_b(self):
        b = np.array([1.0, 2.0, 3.0])
        normalized = fuzzy.normalize_b(b)
        assert np.allclose(normalized, [1 / 6, 2 / 6, 3 / 6])

    def test_normalize_b_zero_sum(self):
        b = np.array([0.0, 0.0])
        normalized = fuzzy.normalize_b(b)
        assert np.allclose(normalized, [0.5, 0.5])


class TestFuzzyComprehensiveEvaluate:
    def test_evaluate_with_scores(self):
        weights = np.array([0.5, 0.5])
        R = np.array([[0.2, 0.8], [0.6, 0.4]])
        result = fuzzy.fuzzy_comprehensive_evaluate(
            weights,
            R,
            fuzzy.FuzzyOperator.WEIGHTED_AVERAGE,
            scores=[60.0, 90.0],
            grades=["差", "好"],
        )
        assert result["operator"] == fuzzy.FuzzyOperator.WEIGHTED_AVERAGE
        assert np.allclose(result["B"], [0.4, 0.6])
        assert np.isclose(result["score"], 0.4 * 60 + 0.6 * 90)
        assert result["grade_index"] == 1
        assert result["grade"] == "好"

    def test_evaluate_without_scores(self):
        weights = np.array([0.5, 0.5])
        R = np.array([[0.2, 0.8], [0.6, 0.4]])
        result = fuzzy.fuzzy_comprehensive_evaluate(weights, R)
        assert np.allclose(result["B"], [0.4, 0.6])
        assert "score" not in result
        assert "grade" not in result

    def test_evaluate_scores_length_mismatch(self):
        weights = np.array([0.5, 0.5])
        R = np.array([[0.2, 0.8], [0.6, 0.4]])
        with pytest.raises(ValueError, match="scores 长度 3 与评语等级数 2 不一致"):
            fuzzy.fuzzy_comprehensive_evaluate(
                weights, R, scores=[60.0, 70.0, 80.0]
            )

    def test_evaluate_grades_length_mismatch(self):
        weights = np.array([0.5, 0.5])
        R = np.array([[0.2, 0.8], [0.6, 0.4]])
        with pytest.raises(ValueError, match="grades 长度 3 与评语等级数 2 不一致"):
            fuzzy.fuzzy_comprehensive_evaluate(
                weights, R, scores=[60.0, 90.0], grades=["差", "中", "好"]
            )


class TestBuildMembershipMatrix:
    def test_build_membership_trapezoid(self):
        # 单个指标，3 个等级，完整边界 [0, 3, 7, 10]
        raw = np.array([[0.0], [5.0], [10.0]])
        R = fuzzy.build_membership_matrix(
            raw,
            boundaries=[0.0, 3.0, 7.0, 10.0],
            method=fuzzy.MembershipMethod.TRAPEZOID,
        )
        assert R.shape == (3, 1, 3)
        # x=0 应完全属于等级 0
        assert np.allclose(R[0, 0], [1.0, 0.0, 0.0])
        # x=5 处于等级 1 的平台区
        assert np.allclose(R[1, 0], [0.0, 1.0, 0.0])
        # x=10 完全属于等级 2
        assert np.allclose(R[2, 0], [0.0, 0.0, 1.0])

    def test_build_membership_piecewise(self):
        raw = np.array([[2.0], [5.0], [8.0]])
        R = fuzzy.build_membership_matrix(
            raw,
            boundaries=[0.0, 3.0, 7.0, 10.0],
            method=fuzzy.MembershipMethod.PIECEWISE,
        )
        assert R.shape == (3, 1, 3)
        assert np.allclose(R[0, 0], [1.0, 0.0, 0.0])
        assert np.allclose(R[1, 0], [0.0, 1.0, 0.0])
        # 8.0 落在最后一个区间 [7, 10]（包含右端点）
        assert np.allclose(R[2, 0], [0.0, 0.0, 1.0])

    def test_build_membership_infer_boundaries(self):
        raw = np.array([[1.0, 10.0], [5.0, 50.0], [9.0, 90.0]])
        # 提供 2 个内部阈值，应自动补充数据最小值和最大值，形成 3 个等级
        R = fuzzy.build_membership_matrix(
            raw,
            boundaries=[4.0, 7.0],
            n_grades=3,
            method=fuzzy.MembershipMethod.PIECEWISE,
        )
        assert R.shape == (3, 2, 3)
        # 第一列完整边界为 [1.0, 4.0, 7.0, 9.0]
        assert np.allclose(R[0, 0], [1.0, 0.0, 0.0])
        assert np.allclose(R[1, 0], [0.0, 1.0, 0.0])
        assert np.allclose(R[2, 0], [0.0, 0.0, 1.0])

    def test_build_membership_with_positive_transform(self):
        raw = np.array([[10.0], [5.0], [0.0]])
        # 极小型指标：数值越小越好，正向化后变为 [0, 5, 10]
        R = fuzzy.build_membership_matrix(
            raw,
            boundaries=[0.0, 5.0, 10.0],
            method=fuzzy.MembershipMethod.PIECEWISE,
            kinds=[2],
        )
        assert R.shape == (3, 1, 2)
        # 原值 10（最差）正向化后为 0，属于等级 0
        assert np.allclose(R[0, 0], [1.0, 0.0])
        # 原值 5 正向化后为 5，属于最后一个等级 1（右端点包含）
        assert np.allclose(R[1, 0], [0.0, 1.0])
        # 原值 0（最优）正向化后为 10，属于等级 1
        assert np.allclose(R[2, 0], [0.0, 1.0])

    def test_build_membership_rejects_non_monotonic_boundaries(self):
        raw = np.array([[1.0], [5.0], [9.0]])
        with pytest.raises(ValueError, match="边界必须严格递增"):
            fuzzy.build_membership_matrix(
                raw, boundaries=[10.0, 5.0, 0.0]
            )

    def test_build_membership_rejects_invalid_boundary_count(self):
        raw = np.array([[1.0], [5.0], [9.0]])
        with pytest.raises(ValueError, match="边界数量应为"):
            fuzzy.build_membership_matrix(
                raw, boundaries=[1.0, 5.0], n_grades=4
            )


class TestInteractiveFunctions:
    def test_prompt_for_operator_weighted_average(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "1")
        op = fuzzy.prompt_for_operator()
        assert op == fuzzy.FuzzyOperator.WEIGHTED_AVERAGE

    def test_prompt_for_operator_bounded_sum(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "4")
        op = fuzzy.prompt_for_operator()
        assert op == fuzzy.FuzzyOperator.BOUNDED_SUM

    def test_prompt_for_weight_source_entropy(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "3")
        source = fuzzy.prompt_for_weight_source()
        assert source == "entropy"

    def test_prompt_for_membership_source_manual(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "1")
        source = fuzzy.prompt_for_membership_source()
        assert source == "manual"

    def test_prompt_for_membership_method_piecewise(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "2")
        method = fuzzy.prompt_for_membership_method()
        assert method == fuzzy.MembershipMethod.PIECEWISE

    def test_read_weights_equal(self, monkeypatch):
        inputs = iter([""])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        weights = fuzzy.read_weights(3)
        assert np.allclose(weights, [1 / 3, 1 / 3, 1 / 3])

    def test_read_weights_manual(self, monkeypatch):
        inputs = iter(["1 2 3"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        weights = fuzzy.read_weights(3)
        assert np.allclose(weights, [1 / 6, 2 / 6, 3 / 6])


class TestEndToEnd:
    def test_end_to_end_single_scheme(self):
        # 一个方案，3 个指标，4 个评语等级
        weights = np.array([0.3, 0.4, 0.3])
        R = np.array([
            [0.1, 0.3, 0.4, 0.2],
            [0.2, 0.4, 0.3, 0.1],
            [0.1, 0.2, 0.4, 0.3],
        ])
        result = fuzzy.fuzzy_comprehensive_evaluate(
            weights,
            R,
            fuzzy.FuzzyOperator.WEIGHTED_AVERAGE,
            scores=[40, 60, 80, 100],
            grades=["差", "中", "良", "优"],
        )
        expected_B = np.array([
            0.3 * 0.1 + 0.4 * 0.2 + 0.3 * 0.1,
            0.3 * 0.3 + 0.4 * 0.4 + 0.3 * 0.2,
            0.3 * 0.4 + 0.4 * 0.3 + 0.3 * 0.4,
            0.3 * 0.2 + 0.4 * 0.1 + 0.3 * 0.3,
        ])
        assert np.allclose(result["B"], expected_B)
        assert np.isclose(result["score"], np.dot(expected_B, [40, 60, 80, 100]))

    def test_end_to_end_auto_membership(self):
        raw = np.array([
            [2.0, 30.0],
            [5.0, 60.0],
            [8.0, 90.0],
        ])
        weights = np.array([0.5, 0.5])
        R_tensor = fuzzy.build_membership_matrix(
            raw,
            boundaries=[0.0, 5.0, 10.0],
            method=fuzzy.MembershipMethod.PIECEWISE,
            kinds=[1, 1],
        )
        assert R_tensor.shape == (3, 2, 2)
        for i in range(3):
            result = fuzzy.fuzzy_comprehensive_evaluate(
                weights,
                R_tensor[i],
                fuzzy.FuzzyOperator.WEIGHTED_AVERAGE,
            )
            assert result["B"].shape == (2,)
            assert np.isclose(np.sum(result["B_normalized"]), 1.0)
