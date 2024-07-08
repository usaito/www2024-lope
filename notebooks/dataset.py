from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import check_random_state

from dataset_coefficients import coef_dict
from utils import (
    sample_action_fast,
    softmax,
    eps_greedy_policy,
    sigmoid
)



def calc_f_x_a(
    x: np.ndarray,
    a_feat: np.ndarray,
    x_coef: np.ndarray,
    a_coef: np.ndarray,
    x_a_coef: np.ndarray
) -> np.ndarray:
    """Expected short-term reward function."""
    f_x_a = (x @ x_coef)[:, np.newaxis, :] + (a_feat @ a_coef)[np.newaxis, :, :]
    for j in range(x_coef.shape[1]):
        f_x_a[:, :, j] += x @ x_a_coef[:, :, j] @ a_feat.T
    mu, sigma = f_x_a.mean(), f_x_a.std()
    f_x_a = (f_x_a - mu) / sigma

    return f_x_a


def calc_h_x_a(
    x: np.ndarray,
    a_feat: np.ndarray,
    x_coef: np.ndarray,
    a_coef: np.ndarray,
    x_a_coef: np.ndarray
) -> np.ndarray:
    """Charicterizes the residual effect that cannot be captured by the short-term rewards."""
    h_x_a = x @ x_coef + (a_feat @ a_coef).T + x @ x_a_coef @ a_feat.T
    mu, sigma = h_x_a.mean(), h_x_a.std()
    h_x_a = (h_x_a - mu) / sigma

    return h_x_a


def calc_g_x_s(
    x: np.ndarray,
    s: np.ndarray,
    x_s_coef: np.ndarray,
    user_clusters: np.ndarray,
) -> np.ndarray:
    """Charicterizes the causal connection between short- and long-term rewards."""
    g_x_s = (s @ x_s_coef)[np.arange(s.shape[0]), user_clusters]

    mu, sigma = g_x_s.mean(), g_x_s.std()
    g_x_s = (g_x_s - mu) / sigma

    return g_x_s


def calc_q_x_a(
    x: np.ndarray,
    a_feat: np.ndarray,
    x_coef_f: np.ndarray,
    a_coef_f: np.ndarray,
    x_a_coef_f: np.ndarray,
    x_coef_h: np.ndarray,
    a_coef_h: np.ndarray,
    x_a_coef_h: np.ndarray,
    x_s_coef: np.ndarray,
    user_clusters: np.ndarray,
    mu: float = None,
    sigma: float = None,
    lambda_: float = 0.5,
    sparsity_factor: float = 0.0,
    reward_type: str = "continuous"
) -> np.ndarray:
    """Expected long-term reward function."""
    h_x_a = calc_h_x_a(x=x, a_feat=a_feat, x_coef=x_coef_h, a_coef=a_coef_h, x_a_coef=x_a_coef_h)
    f_x_a = calc_f_x_a(x=x, a_feat=a_feat, x_coef=x_coef_f, a_coef=a_coef_f, x_a_coef=x_a_coef_f)

    q_x_a = lambda_ * h_x_a
    for a in range(a_feat.shape[0]):
        g_x_s = calc_g_x_s(x=x, s=f_x_a[:, a, :], x_s_coef=x_s_coef, user_clusters=user_clusters)
        q_x_a[:, a] += (1.0 - lambda_) * g_x_s

    if mu is None:
        mu, sigma = q_x_a.mean(), q_x_a.std()
        q_x_a = (q_x_a - mu) / sigma

        if reward_type == "binary":
            q_x_a -= q_x_a.min()
            q_x_a = sigmoid(q_x_a * sparsity_factor)

        return q_x_a, mu, sigma

    else:
        q_x_a = (q_x_a - mu) / sigma

        if reward_type == "binary":
            q_x_a -= q_x_a.min()
            q_x_a = sigmoid(q_x_a * sparsity_factor)

        return q_x_a


def calc_q_x_a_from_s(
    x: np.ndarray,
    s: np.ndarray,
    a_feat: np.ndarray,
    x_coef_h: np.ndarray,
    a_coef_h: np.ndarray,
    x_a_coef_h: np.ndarray,
    x_s_coef: np.ndarray,
    user_clusters: np.ndarray,
    mu: float = None,
    sigma: float = None,
    lambda_: float = 0.5,
    sparsity_factor: float = 0.0,
    reward_type: str = "continuous"
) -> np.ndarray:
    """Expected long-term reward function."""
    h_x_a = calc_h_x_a(x=x, a_feat=a_feat, x_coef=x_coef_h, a_coef=a_coef_h, x_a_coef=x_a_coef_h)

    q_x_a = lambda_ * h_x_a
    for a in range(a_feat.shape[0]):
        g_x_s = calc_g_x_s(x=x, s=s[:, a, :], x_s_coef=x_s_coef, user_clusters=user_clusters)
        q_x_a[:, a] += (1.0 - lambda_) * g_x_s

    if mu is None:
        mu, sigma = q_x_a.mean(), q_x_a.std()
        q_x_a = (q_x_a - mu) / sigma

        if reward_type == "binary":
            q_x_a -= q_x_a.min()
            q_x_a = sigmoid(q_x_a * sparsity_factor)

        return q_x_a, mu, sigma

    else:
        q_x_a = (q_x_a - mu) / sigma

        if reward_type == "binary":
            q_x_a -= q_x_a.min()
            q_x_a = sigmoid(q_x_a * sparsity_factor)

        return q_x_a


@dataclass
class SyntheticDataset:
    n_actions: int
    x_dim: int
    a_dim: int
    s_dim: int
    reward_type: str = "continuous"
    sparsity_factor: float = 0.1
    short_reward_std: float = 0.5
    reward_std: float = 2.0
    lambda_: float = 0.5
    n_all_users : int = 1000
    n_user_clusters: int = 3
    coef_std: float = 1.0
    random_state: int = 12345

    def __post_init__(self) -> None:
        self.random_ = check_random_state(self.random_state)
        self.a_feat = self.random_.normal(size=(self.n_actions, self.a_dim))
        self._generate_user_features()
        self._generate_coefficients()
        self.q_x_a_all, self.mu, self.sigma = self._calc_q_x_a_for_all_users()

    def _generate_user_features(self) -> None:
        self.x_all = self.random_.normal(size=(self.n_all_users, self.x_dim))
        self.user_clusters = KMeans(n_clusters=self.n_user_clusters).fit_predict(self.x_all)

    def _generate_coefficients(self) -> None:
        # coefficients for f_x_a
        self.x_coef_f = self.random_.normal(size=(self.x_dim, self.s_dim), scale=self.coef_std)
        self.a_coef_f = self.random_.normal(size=(self.a_dim, self.s_dim), scale=self.coef_std)
        self.x_a_coef_f = self.random_.normal(size=(self.x_dim, self.a_dim, self.s_dim), scale=self.coef_std)

        # coefficients for h_x_a
        self.x_coef_h = self.random_.normal(size=(self.x_dim, 1), scale=self.coef_std)
        self.a_coef_h = self.random_.normal(size=(self.a_dim, 1), scale=self.coef_std)
        self.x_a_coef_h = self.random_.normal(size=(self.x_dim, self.a_dim), scale=self.coef_std)

        # coefficients for g_x_s
        x_s_dim = 1 + (self.x_dim + self.s_dim)
        x_s_dim += np.arange(1, self.x_dim + self.s_dim + 1).sum()
        self.x_s_coef = coef_dict[self.n_user_clusters]

    def _calc_q_x_a_for_all_users(self) -> np.ndarray:
        q_x_a = calc_q_x_a(
            x=self.x_all,
            a_feat=self.a_feat,
            x_coef_f=self.x_coef_f,
            a_coef_f=self.a_coef_f,
            x_a_coef_f=self.x_a_coef_f,
            x_coef_h=self.x_coef_h,
            a_coef_h=self.a_coef_h,
            x_a_coef_h=self.x_a_coef_h,
            x_s_coef=self.x_s_coef,
            user_clusters=self.user_clusters,
            lambda_=self.lambda_,
            sparsity_factor=self.sparsity_factor,
            reward_type=self.reward_type,
        )

        return q_x_a

    def generate_dataset(
        self,
        n_data: int,
        k: int = 1,
        eps: float = 0.1,
        beta: float = 0.0,
        baseline: bool = True,
    ) -> np.ndarray:
        """Generate a synthetic dataset."""
        data_idx = self.random_.choice(
            self.n_all_users,
            size=n_data,
            replace=True
        )
        x = self.x_all[data_idx]
        f_x_a = calc_f_x_a(
            x=x,
            a_feat=self.a_feat,
            x_coef=self.x_coef_f,
            a_coef=self.a_coef_f,
            x_a_coef=self.x_a_coef_f
        )
        q_x_a = calc_q_x_a(
            x=x,
            a_feat=self.a_feat,
            x_coef_f=self.x_coef_f,
            a_coef_f=self.a_coef_f,
            x_a_coef_f=self.x_a_coef_f,
            x_coef_h=self.x_coef_h,
            a_coef_h=self.a_coef_h,
            x_a_coef_h=self.x_a_coef_h,
            x_s_coef=self.x_s_coef,
            user_clusters=self.user_clusters[data_idx],
            mu=self.mu,
            sigma=self.sigma,
            lambda_=self.lambda_,
            sparsity_factor=self.sparsity_factor,
            reward_type=self.reward_type,
        )

        if baseline:
            pi_0 = softmax(beta * q_x_a)
        else:
            pi_0 = eps_greedy_policy(q_x_a, k, eps)
        actions = sample_action_fast(pi_0, random_state=self.random_state)

        short_term_rewards_ = self.random_.normal(f_x_a, self.short_reward_std)
        short_term_rewards = short_term_rewards_[np.arange(n_data), actions, :]

        q_x_a_factual = q_x_a[np.arange(n_data), actions]
        if self.reward_type == "binary":
            long_term_rewards = self.random_.binomial(n=1, p=q_x_a_factual)
        elif self.reward_type == "continuous":
            long_term_rewards = self.random_.normal(q_x_a_factual, self.reward_std)

        return dict(
            n_data=n_data,
            n_actions=self.n_actions,
            x=x,
            x_s=np.concatenate([x, short_term_rewards], 1),
            a_feat=self.a_feat,
            actions=actions,
            r=long_term_rewards,
            s=short_term_rewards,
            q_x_a=q_x_a,
            q_x_a_factual=q_x_a_factual,
            f_x_a=f_x_a,
            pi_0=pi_0,
            pscore=pi_0[np.arange(n_data), actions],
            new_pi=eps_greedy_policy(q_x_a, k, eps),
        )

    def calc_policy_value(self, q_x_a: np.ndarray, pi: np.ndarray) -> float:
        """Calculate the ground-truth expected long-term reward (performance) of a given policy."""
        return (q_x_a * pi).sum(1).mean()

    def calc_policy_value_beta(self, beta: float = 0.0) -> float:
        pi = softmax(beta * self.q_x_a_all)

        return (self.q_x_a_all * pi).sum(1).mean()

    def calc_policy_value_eps(self, k: int = 1, eps: float = 0.1) -> float:
        pi = eps_greedy_policy(self.q_x_a_all, k, eps)

        return (self.q_x_a_all * pi).sum(1).mean()
