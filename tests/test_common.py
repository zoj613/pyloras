import numpy as np

from pyloras._common import (
    check_random_state,
    safe_random_state,
)


def test_check_random_state():
    rand = np.random.RandomState(12345)
    assert isinstance(check_random_state(rand), np.random.Generator)
    gen = np.random.default_rng(12345)
    assert isinstance(check_random_state(gen), np.random.Generator)


def test_safe_random_state():
    rand = np.random.RandomState(12345)
    assert isinstance(safe_random_state(rand), np.random.RandomState)
    gen = np.random.default_rng(12345)
    assert isinstance(safe_random_state(gen), np.random.RandomState)
