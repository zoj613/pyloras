"""
Copyright (c) 2021, Zolisa Bleki
SPDX-License-Identifier: BSD-3-Clause
"""
from functools import partial
from math import ceil

from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import Substitution
from imblearn.utils._docstring import (
    _random_state_docstring,
    _n_jobs_docstring,
)
from imblearn.utils._validation import check_neighbors_object
from joblib import Parallel, delayed
import numpy as np
from sklearn.manifold import TSNE
from sklearn.utils import check_random_state, parallel_backend


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring,
    n_jobs=_n_jobs_docstring
)
class LORAS(BaseOverSampler):
    """Localized Random Affine Shadowsampling (LoRAS).

    This class implements the LoRAS oversampling technique for imbalanced
    datasets. This technique generates Gaussian noise in small neighborhoods
    around the minority class samples and then the finaly synthetic samples
    are obtained by a convex combination of multipke noisy data points
    (shadowsamples).

    Parameters
    ----------
    {sampling_strategy}
    n_neighbors : int or estimator object, default=None
        If ``int``, number of nearest neighbours to used to construct synthetic
        samples.  If object, an estimator that inherits from
        :class:`~sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.
    n_shadow : int, default=None
        The number of shadow samples to generate per minority class data point.
    std : float or sequence, default=0.005
        The standard deviation of the Normal distribution to add to each
        feature when generating shadow samples. If the input is a sequence, its
        size must be equal to the number of features of ``X`` when calling
        the ``fit_resample`` method. If ``float``, then same standard deviation
        will be used for all shadow samples generated.
    n_affine : int, default=None
        The number of shadow samples to use when generating the synthetic
        samples through random affine combination. If given value must be
        between ``2`` and the number of features using in the fitting data.
        If not given, the value will be set to the number of features in
        fitting data.
    embedding_params : dict, default=None
        A dictionary of additional parameters to pass to the
        :class:`~sklearn.manifold.TSNE` object when creating a 2D manifold of
        the fitting data. The keys are the parameter names and the keys are the
        values. If not given, the default values are used.
    {random_state}
    {n_jobs}

    References
    ----------
    .. [1] Bej, S., Davtyan, N., Wolfien, M. et al. LoRAS: an oversampling
       approach for imbalanced datasets. Mach Learn 110, 279–301 (2021).
       https://doi.org/10.1007/s10994-020-05913-4

    Examples
    --------
    >>> from pyloras import LORAS
    >>> from sklearn.datasets import make_classification
    >>> from collections import Counter
    >>> l = LORAS()
    >>> X, y = make_classification(n_classes=3, class_sep=3,
    ... weights=[0.7, 0.2, 0.1], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=2000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 400, 2: 200, 0: 1400}})
    >>> X_res, y_res = l.fit_resample(X, y)
    >>> print(f"Resampled dataset shape % Counter(y_res))
    Resampled dataset shape Counter({{1: 1200, 2: 1400, 0: 1400}})

    """
    def __init__(
        self,
        *,
        sampling_strategy="auto",
        n_neighbors=None,
        n_shadow=None,
        std=0.005,
        n_affine=None,
        embedding_params=None,
        random_state=None,
        n_jobs=None
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.n_neighbors = n_neighbors
        self.n_shadow = n_shadow
        self.std = std
        self.n_affine = n_affine
        self.embedding_params = embedding_params
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _initialize_params(self, X, y, rng):
        """Initialize the parameter values to their appropriate values."""
        f_size = X.shape[1]
        self.n_affine_ = f_size if self.n_affine is None else self.n_affine
        self.tsne_ = TSNE(n_components=2, n_jobs=self.n_jobs, random_state=rng)

        if self.embedding_params is not None:
            self.tsne_.set_params(**self.embedding_params)

        _, y_counts = np.unique(y, return_counts=True)
        if self.n_neighbors is None:
            n_neighbors = 30 if y_counts.min() >= 100 else 5
        else:
            n_neighbors = self.n_neighbors
        self.nn_ = check_neighbors_object("n_neighbors", n_neighbors)
        if self.n_jobs is not None:
            self.nn_.set_params(n_jobs=self.n_jobs)

        if self.n_affine_ >= self.nn_.n_neighbors * f_size:
            raise ValueError(
                "The number of shadow samples used to construct synthetic "
                "samples must be less than `n_neighbors * number of features`"
            )

        if self.n_shadow is None:
            self.n_shadow_ = max(ceil(2 * f_size / self.nn_.n_neighbors), 40)
        else:
            self.n_shadow_ = self.n_shadow

        try:
            iter(self.std)
            self.std_ = self.std
        except TypeError:
            self.std_ = [self.std] * X.shape[1]

    def _fit_resample(self, X, y):
        X_res = X.copy()
        y_res = y.copy()

        random_state = check_random_state(self.random_state)
        self._initialize_params(X, y, random_state)
        X_embedded = self.tsne_.fit_transform(X_res)
        self.nn_.fit(X_embedded)

        func = partial(
            _make_samples,
            n_shadow=self.n_shadow_,
            std=self.std_,
            n_affine=self.n_affine_,
            random_state=random_state,
            dirichlet_param=[1] * self.n_affine_,
            n_features=X.shape[1],
        )
        loras_samples = []
        loras_classes = []
        for class_sample, n_samples in self.sampling_strategy_.items():
            data_indices = np.flatnonzero(y_res == class_sample)
            # number of synthetic samples per neighborhood group
            n_gen = n_samples // data_indices.shape[0]
            neighborhood_groups = self.nn_.kneighbors(
                X_embedded[data_indices],
                return_distance=False
            )
            with parallel_backend('loky', n_jobs=self.n_jobs):
                samples = Parallel()(
                    delayed(func)(X[i], class_sample, n_gen)
                    for i in neighborhood_groups
                )
            loras_samples.extend(samples)
            loras_classes.extend(
                [class_sample] * n_gen * neighborhood_groups.shape[0]
            )
        X_res = np.vstack((X_res, *loras_samples))
        y_res = np.concatenate((y_res, loras_classes))
        return X_res, y_res


def _make_samples(
    X_neighbors,
    class_sample,
    n_gen,
    n_shadow,
    std,
    n_affine,
    random_state,
    dirichlet_param,
    n_features
):
    """
    Make LoRAS samples given data point and its nearest neighbors,

    This function generates `n_gen` LoRAS synthetic samples given a data point
    and its nearest neighbors in the 2d manifold created by t-SNE.

    X_neighbors : numpy.ndarray
        The neighborhood group (including the data point).
    class_sample : scalar
        The assigned class of the neighborhood group.
    n_gen : int
        The number of synthetic samples to generate.
    n_shadow: int
        The number of shadow samples to use.
    std : list
        Standard deviations per feature of the data point. These values are
        used to sample the Gaussian noise used to create shadow samples.
    n_affine : int
        The size of the subset of shadow samples used to greate the affine
        convex combination of augmented data points.
    random_state : numpy.random.RandomState
        The random number generating object.
    dirichlet_param : list
        A list of 1's whose size is equal to `n_affine`. It is used to
        generate the affine combination weights who follow a Dirichlet
        distribution.
    n_features : int
        Number of features per data point in the neighborhood group.

    """
    size = (n_shadow, X_neighbors.shape[0], n_features)
    shadow_points = random_state.normal(scale=std, size=size)
    shadow_sample = X_neighbors + shadow_points
    shadow_sample = shadow_sample.reshape(
        shadow_sample.shape[0] * shadow_sample.shape[1], -1
    )
    random_index = random_state.randint(
        0, shadow_sample.shape[0], size=(n_gen, n_affine)
    )
    weights = random_state.dirichlet(dirichlet_param, size=n_gen)
    samples = weights[:, None] @ shadow_sample[random_index]
    samples = np.squeeze(samples)
    return samples
