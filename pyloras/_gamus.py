"""
Copyright (c) 2021, Zolisa Bleki
SPDX-License-Identifier: BSD-3-Clause
"""
from collections import defaultdict

from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import Substitution
from imblearn.utils._docstring import (
    _random_state_docstring,
    _n_jobs_docstring,
)
from imblearn.utils._validation import check_neighbors_object
import numpy as np
from sklearn.base import clone
from sklearn.mixture import GaussianMixture

from ._common import check_random_state, safe_random_state


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring,
    n_jobs=_n_jobs_docstring
)
class GAMUS(BaseOverSampler):
    """
    Gaussian Mixture Model based Adaptive Minority up-sampling.

    Parameters
    ----------
    {sampling_strategy}
    n_neighbors : int or estimator object, default=5
        If ``int``, number of nearest neighbours within the minority class per
        minority class data point. If object, an estimator that inherits from
        :class:`~sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the neighbors.
    n_pickedup : int, default=20
        The number of synthetic samples generated for each minority sample and
        neighbor pair. This is equivalent to ``r`` parameter in [1]_.
    n_clusters : int or estimator, default=2
        If ``int``, The number of components used to fit the Gaussian Mixture
        Model on the minority class synthetic data points. If object, an
        estimator from the ``sklearn.mixture`` module that will be used to fit
        the Gaussian Mixture Model.
    gmm_params : dict, default=None
        Non-default parameters to pass to the :class:`~sklearn.mixture.GaussianMixture`
        instance used to fit the Gaussian Mixture Model to the data, if
        `n_cluster` is an integer. This parameter is ignored if `n_cluster` is
        an estimator object.
    proba_threshold : float, default=0.5
        A probability threshold associated with the posterior probability of a
        majority class sample belonging to the i'th Gaussian Mixture cluster.
        If the probability is above the threshold then a majority data point
        will be assigned the cluster associated with the probability.
    weight_threshold : float, default=0.5
        A threshold associated with the proportion of majority samples that are
        assigned to a particular cluster using `proba_threshold`. If the
        proportion of majority samples assigned to a particular cluster is
        above `weight_threshold` then the weight of that cluster is assigned
        `weight_threshold`, else 0.
    {random_state}
    {n_jobs}

    References
    ----------
    .. [1] A. Tripathi, R. Chakraborty and S. K. Kopparapu, "A Novel Adaptive
           Minority Oversampling Technique for Improved Classification in Data
           Imbalanced Scenarios," 2020 25th International Conference on Pattern
           Recognition (ICPR), 2021, pp. 10650-10657,
           doi: 10.1109/ICPR48806.2021.9413002.

    Examples
    --------
    >>> from pyloras import GAMUS
    >>> from sklearn.datasets import make_classification
    >>> from collections import Counter
    >>> l = GAMUS()
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
        n_neighbors=5,
        n_pickedup=10,
        n_clusters=2,
        gmm_params=None,
        proba_threshold=0.5,
        weight_threshold=0.5,
        random_state=None,
        n_jobs=None,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.n_neighbors = n_neighbors
        self.n_pickedup = n_pickedup
        self.n_clusters = n_clusters
        self.gmm_params = gmm_params
        self.proba_threshold = proba_threshold
        self.weight_threshold = weight_threshold
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _validate_estimator(self, rng):
        if not 0 < self.proba_threshold < 1:
            raise ValueError("proba_threshold must be between 0 and 1")
        if not 0 < self.weight_threshold < 1:
            raise ValueError("weight_threshold must be between 0 and 1")

        self.nn_ = check_neighbors_object(
            "n_neighbors", self.n_neighbors, additional_neighbor=1
        )
        self.nn_.set_params(n_jobs=self.n_jobs, algorithm="ball_tree")
        self.alpha_ = np.arange(1, self.n_pickedup + 1) / self.n_pickedup

        if not isinstance(self.n_clusters, int):
            gmm_methods = {'fit', 'predict_proba', 'set_params'}
            for attr in gmm_methods:
                if not hasattr(self.n_clusters, attr):
                    raise ValueError(
                        f'The GMM estimator must implement all of {gmm_methods}'
                    )
            self.gmm_ = clone(self.n_clusters)
            self.gmm_.set_params(random_state=safe_random_state(rng))
        else:
            self.gmm_ = GaussianMixture(
                n_components=self.n_clusters,
                random_state=safe_random_state(rng),
            )
            if self.gmm_params:
                self.gmm_.set_params(**self.gmm_params)

    def _fit_resample(self, X, y):
        random_state = check_random_state(self.random_state)
        self._validate_estimator(random_state)
        X_res = [X.copy()]
        y_res = [y.copy()]

        gamus_samples = defaultdict(lambda: [])
        for minority_class, samples_to_make in self.sampling_strategy_.items():
            if samples_to_make == 0:
                continue

            mask = (y == minority_class)
            Xmin = X[mask]
            self.nn_.fit(Xmin)
            nns = self.nn_.kneighbors(Xmin, return_distance=False)[:, 1:]
            n_hat = []
            # TODO: Vectorize this loop if possible
            for i, point in enumerate(Xmin):
                scaled_diffs = (
                    (point - Xmin[nns[i]])[:, :, None] * self.alpha_
                ).swapaxes(1, 2).reshape(-1, X.shape[1])
                # s_n + alpha * (s_n - s_k) as a 2d array
                n_hat.append(point + scaled_diffs)

            synth_minority = np.concatenate(n_hat)

            self.gmm_.fit(np.concatenate((Xmin, synth_minority)))

            Xmaj = X[~mask]
            q = (self.gmm_.predict_proba(Xmaj) > self.proba_threshold).sum(axis=0)
            weights = 1.0 - q / Xmaj.shape[0]
            # ensure if all majority samples are assigned to one cluster then
            # the weight is set to 1
            weights[q == Xmaj.shape[0]] = 1.
            weights[np.argwhere(weights < self.weight_threshold)] = 0
            self.cluster_weights_ = weights
            samples_per_cluster = np.ceil(
                samples_to_make * weights / sum(weights)
            ).astype(int)
            minority_cluster_labels = (
                np.argmax(self.gmm_.predict_proba(synth_minority), axis=1)
            )
            # TODO: Vectorize this loop if possible
            for i, c in enumerate(samples_per_cluster):
                mat = synth_minority[minority_cluster_labels == i]
                if not c or not mat.shape[0]:
                    continue
                replace = mat.shape[0] < c
                gamus_samples[minority_class].append(
                    random_state.choice(mat, size=c, replace=replace, axis=0)
                )

            samples_to_drop = sum(samples_per_cluster) - samples_to_make
            random_state.shuffle(gamus_samples[minority_class])
            X_res.append(
                np.vstack(gamus_samples[minority_class])[samples_to_drop:]
            )
            y_res.append([minority_class] * X_res[-1].shape[0])

        return np.concatenate(X_res), np.concatenate(y_res)
