import numpy as np
from scipy.stats import rankdata
from sklearn.utils import check_random_state


def sample_action_fast(pi: np.ndarray, random_state: int = 12345) -> np.ndarray:
    """Sampling actions from a given policy quickly."""
    random_ = check_random_state(random_state)
    uniform_rvs = random_.uniform(size=pi.shape[0])[:, np.newaxis]
    cum_pi = pi.cumsum(axis=1)
    flg = cum_pi > uniform_rvs

    return flg.argmax(axis=1)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax function, used when defining a stochastic policy."""
    b = np.max(x, axis=1)[:, np.newaxis]
    numerator = np.exp(x - b)
    denominator = np.sum(numerator, axis=1)[:, np.newaxis]
    return numerator / denominator


def eps_greedy_policy(
    q_x_a: np.ndarray,
    k: int = 1,
    eps: float = 0.1,
) -> np.ndarray:
    """Define an epsilon-greedy policy based on the expected reward function."""
    is_topk = rankdata(-q_x_a, axis=1) <= k
    pi = ((1.0 - eps) / k) * is_topk
    pi += eps / q_x_a.shape[1]
    pi /= pi.sum(1)[:, np.newaxis]

    return pi