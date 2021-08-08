import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import NearestNeighbors

from pyloras import GAMUS


@pytest.fixture
def data():
    return make_classification(n_samples=150, n_features=4, n_informative=4,
                               n_redundant=0, n_repeated=0, n_classes=3,
                               n_clusters_per_class=2,
                               weights=[0.05, 0.05, 0.9],
                               class_sep=0.8, random_state=0)


def test_gamus(data):
    X, y = data
    rng = np.random.RandomState(0)
    gamus = GAMUS(random_state=rng)
    X_res, y_res = gamus.fit_resample(X, y)
    _, y_counts = np.unique(y_res, return_counts=True)
    np.testing.assert_allclose(y_counts[0], y_counts[1])

    assert gamus.nn_.n_neighbors == (5 + 1)
    assert len(gamus.alpha_) ==10
    assert hasattr(gamus, "gmm_")
    assert gamus.gmm_.n_components == 2
    np.testing.assert_allclose(len(gamus.cluster_weights_), 2)

    with pytest.raises(ValueError):
        # test if error is raised when proba_threshold is outside of (0, 1)
        GAMUS(proba_threshold=0.).fit_resample(X, y)

    with pytest.raises(ValueError):
        # test if error is raised when weight_threshold is outside of (0, 1)
        GAMUS(weight_threshold=1.).fit_resample(X, y)

    gamus = GAMUS(gmm_params={'init_params': 'random'})
    X_res2, y_res2 = gamus.fit_resample(X, y)
    assert gamus.gmm_.init_params == 'random'

    gmm = BayesianGaussianMixture(n_components=10)
    gamus = GAMUS(n_clusters=gmm, gmm_params={'init_params': 'random'})
    X_res2, y_res2 = gamus.fit_resample(X, y)
    assert gamus.gmm_.n_components == 10
    # ensure a copy of the object was made
    assert gmm is not gamus.gmm_
    # ensure ``gmm_params`` is ignored if n_clusters is an object
    assert gamus.gmm_.init_params != 'random'

    with pytest.raises(ValueError, match='The GMM estimator must implement'):
        class WrongGMM(BaseEstimator):
            def __init__(self, n_components=None):
                self.n_components = None
        GAMUS(n_clusters=WrongGMM()).fit_resample(X, y)


    # test reporducibility
    gamus = GAMUS(random_state=np.random.RandomState(0))
    X_res2, y_res2 = gamus.fit_resample(X, y)
    assert np.allclose(X_res, X_res2)
    assert np.allclose(y_res, y_res2)

    # test if no samples are added for an already balanced dataset.
    y = np.repeat([0, 1, 2], 50)
    X_res, y_res = gamus.fit_resample(X, y)
    np.testing.assert_allclose(X_res, X)
    np.testing.assert_allclose(y_res, y)
