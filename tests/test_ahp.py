import src.algorithms.ahp as ahp
from src.algorithms.ahp import WeightVectorType
import numpy as np


SAMPLE_MATRIX = np.array([
    [1, 2, 3],
    [1 / 2, 1, 4],
    [1 / 3, 1 / 4, 1]
])


def test_calculate_weight_vector():
    n = SAMPLE_MATRIX.shape[0]
    ans_val, ans_vec = ahp.calculate_weight_vector(SAMPLE_MATRIX, n, WeightVectorType.EIGVEC)
    assert isinstance(ans_val, (float, np.floating))
    assert ans_vec.shape == (n,)


def test_calculate_weights(capsys):
    ahp.calculate_weights(SAMPLE_MATRIX, WeightVectorType.EIGVEC)
    captured = capsys.readouterr()
    assert "CR为" in captured.out


def test_prompt_for_method(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "3")
    method = ahp.prompt_for_method()
    assert method == WeightVectorType.EIGVEC


def test_prompt_for_method_defaults_to_eigvec(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")
    method = ahp.prompt_for_method()
    assert method == WeightVectorType.EIGVEC


def test_prompt_for_method_rejects_invalid_then_accepts(monkeypatch):
    inputs = iter(["x", "2"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    method = ahp.prompt_for_method()
    assert method == WeightVectorType.GEOAVE
