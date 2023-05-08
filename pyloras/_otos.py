from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import Substitution
from imblearn.utils._docstring import (
    _random_state_docstring,
    _n_jobs_docstring,
)
from sklearn.svm import LinearSVC
import numpy as np

from ._common import check_random_state, safe_random_state


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring,
    n_jobs=_n_jobs_docstring
)
class OTOS(BaseOverSampler):
    def __init__(
        self,
        *,
        sampling_strategy="auto",
        svc_reg=1.0,
        ot_reg=1.0,
        tradeoff=1.0,
        random_state=None,
        max_iter=100,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.svc_reg = svc_reg
        self.ot_reg = ot_reg
        self.tradeoff = tradeoff
        self.random_state = random_state
        self.max_iter = max_iter

    def _fit_resample(self, X, y):
        import ot
        random_state = check_random_state(self.random_state)
        X_res = [X.copy()]
        y_res = [y.copy()]
        svc = LinearSVC(
            loss="hinge", C=self.svc_reg, random_state=safe_random_state(random_state)
        )
        for minority_class, samples_to_make in self.sampling_strategy_.items():
            if samples_to_make == 0:
                continue
            X_p = X[y == minority_class]
            X_n = X[y != minority_class]
            n_p = X_p.shape[0]
            n_n = X_n.shape[0]
            n_r = samples_to_make
            one_r = np.ones((n_r, 1))
            one_n = np.ones((n_n, 1))
            # set initial distribution for mu_r and mu_p
            mu_r = np.asarray([1.0 / n_r] * n_r)
            mu_p = np.asarray([1.0 / n_p] * n_p)
            T = mu_r[:, None] @ mu_p[:, None].T
            # manufactor a binary classification problem
            _y = np.empty_like(y)
            _y[y == minority_class] = 0
            _y[y != minority_class] = 1
            svc.fit(X, _y)
            w = svc.coef_.T

            hingelosses = np.concatenate(
                [
                    np.atleast_1d(max(1 - y_i * svc.coef_ @ x_row, 0.0))
                    for y_i, x_row in zip(y[y == minority_class], X_p)
                ]
            )
            mu_p = np.exp(hingelosses)
            mu_p /= mu_p.sum()

            D_r = np.diag(1 / mu_r)
            X_r = D_r @ T @ X_p
            # C_p = np.apply_along_axis(c_row, axis=-1, arr=X_r)
            C_p = np.asarray(
                [
                    [np.linalg.norm(x_row - row) for row in X_p]
                    for x_row in X_r
                ]
            )
            wwT = w @ w.T
            Theta = self.tradeoff * C_p.T - X_p @ np.kron(one_r.T, wwT @ X_n.T @ one_n + n_n * w) @ D_r
            Phi = X_p @ wwT @ X_p.T
            Psi = D_r.T @ D_r

            for _ in range(self.max_iter):
                transport_cost = Theta.T + n_n * Psi @ T @ Phi
                T = ot.sinkhorn(mu_r, mu_p, transport_cost, self.ot_reg)
            X_res.append(D_r @ T @ X_p)
            y_res.append([minority_class] * n_r)

        return np.concatenate(X_res), np.concatenate(y_res)
