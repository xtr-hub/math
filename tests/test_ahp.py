import src.ahp as ahp
from src.ahp import WeightVectorType
import numpy as np
import pytest
from src.data import save_processed
from src.data import load_raw

def test_calaulate_weights_vectors():
    matrix = np.array(load_raw("judgment_matrix.csv"))
    ahp.calaulate_weights_vectors(matrix, WeightVectorType.EIGVEC)