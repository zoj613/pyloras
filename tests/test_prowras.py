import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors

from pyloras import ProWRAS


@pytest.fixture
def data():
    return make_classification(n_samples=150, n_features=4, n_informative=4,
                               n_redundant=0, n_repeated=0, n_classes=3,
                               n_clusters_per_class=2,
                               weights=[0.01, 0.05, 0.94],
                               class_sep=0.8, random_state=0)


def test_prowras(data):
    X, y = data
    rng = np.random.RandomState(0)
    lrs = ProWRAS(random_state=rng)
    X_res, y_res = lrs.fit_resample(X, y)
    _, y_counts = np.unique(y_res, return_counts=True)
    np.testing.assert_allclose(y_counts[0], y_counts[1])

    assert len(lrs.cdata_[0].clusters) < lrs.max_clusters
    assert len(lrs.cdata_[1].clusters) < lrs.max_clusters
    assert max(lrs.cdata_[0].n_affine) <= lrs.max_affine
    assert max(lrs.cdata_[1].n_affine) <= lrs.max_affine

    # test reporducibility
    lrs = ProWRAS(random_state=np.random.RandomState(0))
    X_res2, y_res2 = lrs.fit_resample(X, y)
    assert np.allclose(X_res, X_res2)
    assert np.allclose(y_res, y_res2)

    nn = NearestNeighbors()
    std =  [0.001] * X.shape[1]
    lrs = ProWRAS(
        n_neighbors_max=nn, random_state=np.random.RandomState(0), std=std
    )
    X_res2, y_res2 = lrs.fit_resample(X, y)
    assert np.allclose(X_res, X_res2)
    assert np.allclose(y_res, y_res2)

    # test wrong std size
    std = [0.001] * 2
    with pytest.raises(ValueError):
        ProWRAS(std=std).fit_resample(X, y)

    # test if no samples are added for an already balanced dataset.
    y = np.repeat([0, 1, 2], 50)
    X_res, y_res = lrs.fit_resample(X, y)
    np.testing.assert_allclose(X_res, X)
    np.testing.assert_allclose(y_res, y)
    assert not lrs.cdata_
