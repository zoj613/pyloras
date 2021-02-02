import pytest
import numpy as np
from sklearn.neighbors import NearestNeighbors

from pyloras import LORAS


@pytest.fixture
def data():
    X = np.array([[-1.713043,  1.811711,  0.496868, -1.891984,  1.877141],
                  [ 1.828116,  0.207981, -1.318674, -0.551289,  1.604962],
                  [ 1.396854, -0.793973,  0.494784,  0.718366,  0.805485],
                  [ 0.903668, -0.7054  ,  1.954668, -0.594842, -0.74627 ],
                  [ 1.488443, -0.176546,  0.100354,  0.477952, -0.864051],
                  [ 0.332138, -1.267381,  0.872687,  1.297604,  1.516713],
                  [ 1.838971,  1.862792, -1.837891, -0.630398, -1.920483],
                  [ 0.070929,  0.36782 , -1.168828, -0.405528,  1.378379],
                  [-0.220063, -1.925132, -0.260998, -1.167207, -1.90441 ],
                  [-1.549394, -0.094323, -1.290381,  0.567559, -0.281728],
                  [ 0.292762, -0.940415, -1.948312,  0.983617, -1.801541],
                  [-1.760325,  1.122002, -1.122498,  1.553093,  0.374005],
                  [-0.787668, -1.943021, -0.729356,  1.606771,  0.385803],
                  [ 0.31696 , -1.540417, -0.33592 ,  0.599856,  0.652828],
                  [-0.285518, -1.899701, -0.861786, -0.616594, -0.111652],
                  [-0.852885,  1.450134, -1.636573, -0.232598,  0.669851],
                  [-0.682753,  1.630959, -1.831675, -0.480502, -1.930837],
                  [ 0.357201,  1.719938, -0.742491, -1.761611,  0.26162 ],
                  [-1.206372, -0.214496, -0.852919, -0.00802 ,  0.033227],
                  [ 0.955751,  0.750906, -1.244543,  0.632058, -1.315618],
                  [-0.496111, -1.908566,  1.399542, -0.419715, -0.876355],
                  [-0.3267  , -0.552419, -0.570479,  0.913295, -0.076761],
                  [-1.139276,  0.6482  , -1.932879,  1.082885, -0.797141],
                  [-1.159733, -1.820455,  1.548121,  1.310965,  1.422993],
                  [-0.492523,  1.044069,  1.047334, -1.062792,  1.598441],
                  [-0.813246,  1.633826, -0.717498, -1.362327, -1.104209],
                  [-1.125088, -1.095099,  0.719081, -0.095707,  0.39845 ],
                  [-0.416416, -0.900493, -0.691456,  0.241477,  0.722071],
                  [ 1.22531 ,  1.448872, -1.883669,  0.993742,  1.941489],
                  [-1.0524  , -1.067037, -1.996986,  0.336446, -0.725008]])

    y = np.array([1., 0, 1., 1., 0., 1., 1., 0., 0, 0., 0., 0., 0., 1., 0.,
                  0., 0, 0., 0., 1., 0., 1., 0., 0, 1., 0., 0., 0., 0., 0.])
    return X, y


def test_loras(data):
    X, y = data
    rng = np.random.RandomState(0)
    lrs = LORAS(random_state=rng)
    X_res, y_res = lrs.fit_resample(X, y)
    # test initialized parameter values
    assert lrs.n_affine_ == 5
    assert lrs.nn_.n_neighbors == 5
    assert lrs.n_shadow_ == 40
    assert isinstance(lrs.std_, list)
    assert np.allclose(lrs.std_, 0.005)
    assert len(np.argwhere(y_res == 1)) == 18

    # test reporducibility
    lrs = LORAS(random_state=np.random.RandomState(0))
    X_res2, y_res2 = lrs.fit_resample(X, y)
    assert np.allclose(X_res, X_res2)
    assert np.allclose(y_res, y_res2)

    nn = NearestNeighbors(n_jobs=3)
    std = [0.5, 1, 0.3, 0.1, 0.9]
    lrs = LORAS(n_neighbors=nn, random_state=rng, std=std)
    X_res, y_res = lrs.fit_resample(X, y)
    assert np.allclose(lrs.std_, std)
    assert lrs.nn_.n_jobs == 3
    lrs = LORAS(n_neighbors=nn, random_state=rng, n_jobs=4, n_shadow=3)
    X_res, y_res = lrs.fit_resample(X, y)
    assert lrs.nn_.n_jobs == 4
    assert lrs.n_shadow_ == 3

    with pytest.raises(ValueError):
        LORAS(n_affine=500).fit_resample(X, y)

    lrs = LORAS(random_state=rng, embedding_params={'n_iter': 270})
    X_res, y_res = lrs.fit_resample(X, y)
    assert lrs.tsne_.n_iter == 270
