import numpy as np


def check_random_state(random_state):
    if isinstance(random_state, np.random.RandomState):
        return np.random.default_rng(random_state._bit_generator)
    return np.random.default_rng(random_state)


def safe_random_state(random_state):
    """
    To be used with sklearn Estimators that expect a random_state argument.

    This safely passes the required ``np.random.RandomState`` instance to the
    estimator when the user has a ``np.random.Generator`` instead.
    """
    if isinstance(random_state, np.random.Generator):
        return np.random.RandomState(random_state._bit_generator)
    return random_state
