import ot
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import Substitution
from imblearn.utils._docstring import (
    _random_state_docstring,
    _n_jobs_docstring,
)
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
        svm_regularization=1.0,
        ot_regularization=1.0,
        tradeoff=1.0,
        random_state=None,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.svm_regularization = svm_regularization
        self.ot_regularization = ot_regularization
        self.tradeoff = tradeoff
        self.random_state = random_state

    def fit_resample(self, X, y):
        return X, y
