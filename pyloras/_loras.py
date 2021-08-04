"""
Copyright (c) 2021, Zolisa Bleki
SPDX-License-Identifier: BSD-3-Clause
"""
from collections import defaultdict
from math import ceil

from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import Substitution
from imblearn.utils._docstring import (
    _random_state_docstring,
    _n_jobs_docstring,
)
from imblearn.utils._validation import check_neighbors_object
import numpy as np
from sklearn.manifold import TSNE
from sklearn.base import clone

from ._common import check_random_state, safe_random_state



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
    are obtained by a convex combination of multiple noisy data points
    (shadowsamples).

    Parameters
    ----------
    {sampling_strategy}
    n_neighbors : int or estimator object, default=None
        If ``int``, number of nearest neighbours to used to construct synthetic
        samples. If object, an estimator that inherits from
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
        samples through random affine combinations. If given, the value must be
        between ``2`` and the number of features used in the fitting data.
        If not given, the value will be set to the total number of features in
        fitting data.
    manifold_learner : object, default=None
        An instance of an object that to perform a 2-dimensional embedding of
        a dataset. It must implement the scikit-learn Estimator interface,
        ``fit_transform`` and ``set_params`` methods must be implemented.
        If not given, the :class:`~sklearn.manifold.TSNE` class is used to
        obtain the 2d manifold of the data. Defaults to None.
    manifold_learner_params : dict, default=None
        A dictionary of additional parameters to pass to the instance of the
        ``manifold_learner`` (or TSNE if ``manifold_learner`` is None) when
        creating a 2D manifold of the fitting data. The keys are the parameter
        names and the values are the values. If not given, the default values
        are used.
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
    Resampled dataset shape Counter({{1: 1400, 2: 1400, 0: 1400}})

    """
    def __init__(
        self,
        *,
        sampling_strategy="auto",
        n_neighbors=None,
        n_shadow=None,
        std=0.005,
        n_affine=None,
        manifold_learner=None,
        manifold_learner_params=None,
        random_state=None,
        n_jobs=None
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.n_neighbors = n_neighbors
        self.n_shadow = n_shadow
        self.std = std
        self.n_affine = n_affine
        self.manifold_learner = manifold_learner
        self.manifold_learner_params = manifold_learner_params
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _check_2d_manifold_learner(self):
        if (not hasattr(self.manifold_learner, "fit_transform") or
                not hasattr(self.manifold_learner, "set_params")):
            raise ValueError(
                "The 2d manifold learner must implement the ``fit_transform`` "
                "and ``set_params`` methods"
            )

        return clone(self.manifold_learner)

    def _initialize_params(self, X, y, rng):
        """Initialize the parameter values to their appropriate values."""
        f_size = X.shape[1]
        self.n_affine_ = f_size if self.n_affine is None else self.n_affine

        if self.manifold_learner:
            self.manifold_learner_ = self._check_2d_manifold_learner()
        else:
            self.manifold_learner_ = TSNE(n_components=2)
        if self.manifold_learner_params is not None:
            self.manifold_learner_.set_params(**self.manifold_learner_params)
        try:
            self.manifold_learner_.set_params(random_state=safe_random_state(rng))
        except ValueError:
            pass

        _, y_counts = np.unique(y, return_counts=True)
        if self.n_neighbors is None:
            n_neighbors = 30 if y_counts.min() >= 100 else 5
        else:
            n_neighbors = self.n_neighbors
        self.nn_ = check_neighbors_object("n_neighbors", n_neighbors)
        if self.n_jobs is not None:
            self.nn_.set_params(n_jobs=self.n_jobs)

        if self.n_shadow is None:
            self.n_shadow_ = max(ceil(2 * f_size / self.nn_.n_neighbors), 40)
        else:
            self.n_shadow_ = self.n_shadow

        if self.n_affine_ >= self.nn_.n_neighbors * self.n_shadow_:
            raise ValueError(
                "The number of shadow samples used to create an affine random "
                "combination must be less than `n_neighbors * n_shadow`."
            )

        try:
            iter(self.std)
            self.std_ = self.std
        except TypeError:
            self.std_ = [self.std] * f_size

    def _fit_resample(self, X, y):
        random_state = check_random_state(self.random_state)
        self._initialize_params(X, y, random_state)
        n_features = X.shape[1]

        X_res = [X.copy()]
        y_res = [y.copy()]
        dirichlet_param = [1] * self.n_affine_
        loras_samples = defaultdict(lambda : [])

        for minority_class, samples_to_make in self.sampling_strategy_.items():
            if samples_to_make == 0:
                continue
            X_minority = X[y == minority_class]
            X_embedded = self.manifold_learner_.fit_transform(X_minority)
            self.nn_.fit(X_embedded)
            neighborhoods = self.nn_.kneighbors(X_embedded, return_distance=False)
            num_loras = ceil(samples_to_make / X_embedded.shape[0])
            for neighbor_group in neighborhoods:
                shadow_sample_size = (self.n_shadow_, self.nn_.n_neighbors, n_features)
                total_shadow_samples = (
                    X_minority[neighbor_group] +
                    random_state.normal(scale=self.std_, size=shadow_sample_size)
                ).reshape(self.n_shadow_ * self.nn_.n_neighbors, n_features)
                random_index = random_state.integers(
                    0, total_shadow_samples.shape[0], size=(num_loras, self.n_affine_)
                )
                weights = random_state.dirichlet(dirichlet_param, size=num_loras)
                loras_samples[minority_class].append(
                    (weights[:, None] @ total_shadow_samples[random_index])
                    .reshape(num_loras, n_features)
                )
            # keep only ``samples_to_make`` synthetic samples from the generated.
            samples_to_drop = X_embedded.shape[0] * num_loras - samples_to_make
            random_state.shuffle(loras_samples[minority_class])
            X_res.append(
                np.concatenate(loras_samples[minority_class])[samples_to_drop:]
            )
            y_res.append([minority_class] * samples_to_make)

        return np.concatenate(X_res), np.concatenate(y_res)
