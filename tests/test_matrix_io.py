import numpy as np
import pytest

import src.io.matrix_io as matrix_io


SAMPLE_MATRIX = np.array([
    [1, 2, 3],
    [1 / 2, 1, 4],
    [1 / 3, 1 / 4, 1]
])


def test_read_int(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "5")
    assert matrix_io.read_int("prompt: ") == 5


def test_read_int_with_range_rejects_then_accepts(monkeypatch):
    inputs = iter(["1", "10", "5"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    assert matrix_io.read_int("prompt: ", min_value=2, max_value=8) == 5


def test_read_int_rejects_non_integer(monkeypatch):
    inputs = iter(["x", "3"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    assert matrix_io.read_int("prompt: ") == 3


def test_read_floats(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "1 2 3")
    assert matrix_io.read_floats("prompt: ") == [1.0, 2.0, 3.0]


def test_read_floats_with_count_and_positive(monkeypatch):
    inputs = iter(["-1 2 3", "0 2 3", "1 2 3"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    assert matrix_io.read_floats("prompt: ", count=3, positive=True) == [1.0, 2.0, 3.0]


def test_read_reciprocal_matrix(monkeypatch):
    inputs = iter(["3", "2 3", "4"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    matrix = matrix_io.read_reciprocal_matrix()
    assert np.allclose(matrix, SAMPLE_MATRIX)


def test_read_matrix(monkeypatch):
    inputs = iter(["1 2 3", "4 5 6"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    matrix = matrix_io.read_matrix(2, 3)
    expected = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.allclose(matrix, expected)

def test_read_ints(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "1 2 3")
    assert matrix_io.read_ints("prompt: ") == [1, 2, 3]


def test_read_matrix_rejects_wrong_count_then_accepts(monkeypatch):
    inputs = iter(["1 2", "1 2 3", "4 5 6"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    matrix = matrix_io.read_matrix(2, 3)
    expected = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.allclose(matrix, expected)


def test_print_matrix(capsys):
    matrix_io.print_matrix(SAMPLE_MATRIX, title="测试矩阵：")
    captured = capsys.readouterr()
    assert "测试矩阵：" in captured.out
    assert "1." in captured.out
