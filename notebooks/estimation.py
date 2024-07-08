from typing import List

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor



def run_long_term_experiment(D_E_list: List[dict]):
    """Estimate the long-term expected reward under a given policy via actually running a long-term experiment."""
    estimated_values = dict()
    for i, D_E in enumerate(D_E_list):
        estimated_values[i] = D_E["r"].mean()
    return estimated_values


def run_long_term_ci(
    D_H: dict, 
    D_E_list: List[dict], 
    g_x_s_model = MLPRegressor(hidden_layer_sizes=(10, 10, 10), random_state=12345),
) -> float:
    """Estimate the long-term expected reward under a given policy via long-term causal inference based on the surrogacy assumption."""
    estimated_values = dict()
    g_x_s_model.fit(D_H["x_s"], D_H["r"])
    for i, D_E in enumerate(D_E_list):
        estimated_values[i] = g_x_s_model.predict(D_E["x_s"]).mean()
    
    return estimated_values


def estimate_q_x_a_via_regression(
    D_H: dict, q_x_a_model = MLPRegressor(hidden_layer_sizes=(10, 10, 10), random_state=12345)
) -> np.ndarray:
    """Estimate the expected reward function (`q(x,a)`), which is used for DR."""
    n_data, n_actions = D_H["n_data"], D_H["n_actions"]
    x, r = D_H["x"], D_H["r"]
    actions, a_feat = D_H["actions"], D_H["a_feat"]

    x_a = np.concatenate([x, a_feat[actions]], 1)
    q_x_a_model.fit(x_a, r)
    
    q_x_a_hat = np.zeros((n_data, D_H["n_actions"]))
    for a in range(n_actions):
        x_a = np.concatenate([x, np.tile(a_feat[a], (n_data, 1))], 1)
        q_x_a_hat[:, a] = q_x_a_model.predict(x_a)
    
    return q_x_a_hat


def run_typical_ope(
    D_H: dict,
    q_x_a_model = MLPRegressor(hidden_layer_sizes=(10, 10, 10), random_state=12345),
) -> float:
    """Estimate the long-term expected reward under a given policy via typical OPE (IPS and DR), which does not utilize short-term rewards."""
    n_data = D_H["n_data"]
    r, actions = D_H["r"], D_H["actions"]
    pi_0 = D_H["pi_0"]
    q_x_a_hat = estimate_q_x_a_via_regression(D_H, q_x_a_model=q_x_a_model)
    
    estimated_values = dict()
    factual_q_x_a_hat = q_x_a_hat[np.arange(n_data), actions]
    for i, policy in enumerate(["pi_0", "new_pi"]):
        pi_i = D_H[policy]
        iw = pi_i[np.arange(n_data), actions] / pi_0[np.arange(n_data), actions]
        estimated_values[f"ips{i}"] = (iw * r).mean()
        
        q_x_pi_hat = (q_x_a_hat * pi_i).sum(1)
        estimated_values[f"dr{i}"] = (iw * (r - factual_q_x_a_hat) + q_x_pi_hat).mean()
    
    return estimated_values


def run_long_term_ope(
    D_H: dict, D_E_0: dict,
    q_x_a_model = MLPRegressor(hidden_layer_sizes=(10, 10, 10), random_state=12345),
    pi_a_x_s_model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), random_state=12345),
) -> float:
    """Estimate the long-term expected reward under a given policy via long-term OPE (ours), which combines short- and long-term rewards in the historical data."""
    n_data = D_H["n_data"]
    x_s, r, actions = D_H["x_s"], D_H["r"], D_H["actions"]
    pi_0 = D_H["pi_0"]
    q_x_a_hat = estimate_q_x_a_via_regression(D_H, q_x_a_model=q_x_a_model)
    
    x_s_ = np.concatenate([D_H["x_s"], D_E_0["x_s"]])
    actions_ = np.concatenate([actions, D_E_0["actions"]])
    observed_action_set = np.unique(actions_)
    pi_a_x_s_model.fit(x_s_, actions_)
    pi_a_x_s_hat = np.zeros(shape=(n_data, D_H["n_actions"]))
    pi_a_x_s_hat[:, observed_action_set] = pi_a_x_s_model.predict_proba(x_s)
    
    estimated_values = dict()
    factual_q_x_a_hat = q_x_a_hat[np.arange(n_data), actions]
    for i, policy in enumerate(["pi_0", "new_pi"]):
        pi_i = D_H[policy]
        iw_hat = ((pi_i / pi_0) * pi_a_x_s_hat).sum(1)
        q_x_pi_hat = (q_x_a_hat * pi_i).sum(1)
        estimated_values[i] = (iw_hat * (r - factual_q_x_a_hat) + q_x_pi_hat).mean()
    
    return estimated_values


def run_all(D_H: dict, D_E_list: List[dict]) -> List[dict]:
    """Run all methods to estimate the long-term expected reward under a given policy simultaneously."""
    long_term_experiment = run_long_term_experiment(D_E_list)
    long_term_ci = run_long_term_ci(D_H, D_E_list)
    typical_ope = run_typical_ope(D_H)
    long_term_ope = run_long_term_ope(D_H, D_E_list[0])
    
    estimated_values_of_baseline = {
        "long_term_experiment": long_term_experiment[0],
        "long_term_ci": long_term_ci[0],
        "typical_ope_ips": typical_ope["ips0"],
        "typical_ope_dr": typical_ope["dr0"],
        "long_term_ope": long_term_ope[0]
    }
    estimated_values_of_new_policy = {
        "long_term_experiment": long_term_experiment[1],
        "long_term_ci": long_term_ci[1],
        "typical_ope_ips": typical_ope["ips1"],
        "typical_ope_dr": typical_ope["dr1"],
        "long_term_ope": long_term_ope[1]
    }
    estimated_policy_comparison = dict()
    for method in estimated_values_of_new_policy:
        is_new_policy_better = np.int(estimated_values_of_baseline[method] < estimated_values_of_new_policy[method])
        estimated_policy_comparison[method] = is_new_policy_better - (1 - is_new_policy_better)
    
    return estimated_values_of_baseline, estimated_values_of_new_policy, estimated_policy_comparison