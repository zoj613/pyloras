import numpy as np
import pytest
from sklearn.datasets import make_classification

from pyloras import OTOS


@pytest.fixture
def data():
    return make_classification(n_samples=150, n_features=4, n_informative=4,
                               n_redundant=0, n_repeated=0, n_classes=3,
                               n_clusters_per_class=2,
                               weights=[0.01, 0.05, 0.94],
                               class_sep=0.8, random_state=0)


def test_otos(data):
    X, y = data
    rng = np.random.RandomState(12345)
    otos = OTOS(random_state=rng)
    X_res, y_res = otos.fit_resample(X, y)
    _, y_counts = np.unique(y_res, return_counts=True)
    np.testing.assert_allclose(y_counts[0], y_counts[1])
