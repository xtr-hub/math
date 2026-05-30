import src.ahp as ahp
from src.ahp import WeightVectorType
import numpy as np
import pytest
from src.data import save_processed
from src.data import load_raw

def test_calaulate_weights_vectors():
    matrix = load_raw("judgment_matrix.xlsx").iloc[:, 1:].to_numpy()
    ans_val, ans_vec = ahp.calculate_weight_vector(matrix, WeightVectorType.EIGVEC)
    print(ans_val, ans_vec)