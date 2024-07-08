from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.utils import check_random_state
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from estimation import estimate_q_x_a_via_regression
from utils import softmax




@dataclass
class RegBasedPolicyLearner:
    """Typical regression-based baseline."""

    q_x_a_model = MLPRegressor(
        hidden_layer_sizes=(30, 30, 30), random_state=12345
    )
    random_state: int = 12345

    def fit(self, D_H: dict) -> None:

        x, r = D_H["x"], D_H["r"]
        actions, a_feat = D_H["actions"], D_H["a_feat"]

        x_a = np.concatenate([x, a_feat[actions]], 1)
        self.q_x_a_model.fit(x_a, r)

    def predict(self, D_test: np.ndarray, tau: float = 0.01) -> np.ndarray:

        n_data, n_actions = D_test["n_data"], D_test["n_actions"]
        x, a_feat = D_test["x"], D_test["a_feat"]
        q_x_a_hat = np.zeros((n_data, n_actions))
        for a in range(n_actions):
            x_a = np.concatenate([x, np.tile(a_feat[a], (n_data, 1))], 1)
            q_x_a_hat[:, a] = self.q_x_a_model.predict(x_a)

        return softmax(q_x_a_hat / tau)
    
    
    
@dataclass
class NNPolicyDataset(torch.utils.data.Dataset):
    """PyTorch dataset for NNPolicyLearner"""

    context: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    pscore: np.ndarray
    q_x_a_hat: np.ndarray
    pi_0: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert (
            self.context.shape[0]
            == self.action.shape[0]
            == self.reward.shape[0]
            == self.pscore.shape[0]
            == self.q_x_a_hat.shape[0]
            == self.pi_0.shape[0]
        )

    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
            self.pscore[index],
            self.q_x_a_hat[index],
            self.pi_0[index],
        )

    def __len__(self):
        return self.context.shape[0]

    

    
@dataclass
class TypicalOPL:
    """IPS-PG and DR-PG (typical model-free baselines) which use only long-term rewards in the historical data."""

    x_dim: int
    n_actions: int
    hidden_layer_size: tuple = (30, 30, 30)
    activation: str = "elu"
    batch_size: int = 32
    learning_rate_init: float = 0.005
    alpha: float = 1e-6
    log_eps: float = 1e-10
    solver: str = "adam"
    max_iter: int = 200
    off_policy_objective: str = "ips"
    random_state: int = 12345

    def __post_init__(self) -> None:
        """Initialize class."""
        layer_list = []
        input_size = self.x_dim

        if self.activation == "tanh":
            activation_layer = nn.Tanh
        elif self.activation == "relu":
            activation_layer = nn.ReLU
        elif self.activation == "elu":
            activation_layer = nn.ELU

        for i, h in enumerate(self.hidden_layer_size):
            layer_list.append(("l{}".format(i), nn.Linear(input_size, h)))
            layer_list.append(("a{}".format(i), activation_layer()))
            input_size = h
        layer_list.append(("output", nn.Linear(input_size, self.n_actions)))
        layer_list.append(("softmax", nn.Softmax(dim=1)))

        self.nn_model = nn.Sequential(OrderedDict(layer_list))

        self.random_ = check_random_state(self.random_state)
        self.train_loss = []
        self.train_value = []
        self.test_value = []

    def fit(self, D_H: dict, D_test: dict) -> None:
        x, a, r = D_H["x"], D_H["actions"], D_H["r"]
        pscore, pi_0 = D_H["pscore"], D_H["pi_0"]
        if self.off_policy_objective == "ips":
            q_x_a_hat = np.zeros((r.shape[0], self.n_actions))
        elif self.off_policy_objective == "dr":
            q_x_a_hat = estimate_q_x_a_via_regression(D_H)

        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.AdamW(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")

        training_data_loader = self._create_train_data_for_opl(
            x, a, r, pscore, q_x_a_hat, pi_0,
        )

        # start policy training
        x_test = D_test["x"]
        q_x_a_train, q_x_a_test = D_H["q_x_a"], D_test["q_x_a"]
        for _ in range(self.max_iter):
            pi_train = self.predict(D_H)
            self.train_value.append((q_x_a_train * pi_train).sum(1).mean())
            pi_test = self.predict(D_test)
            self.test_value.append((q_x_a_test * pi_test).sum(1).mean())

            loss_epoch = 0.0
            self.nn_model.train()
            for x, a, r, p, q_x_a_hat_, pi_0_ in training_data_loader:
                optimizer.zero_grad()
                pi = self.nn_model(x)
                loss = -self._estimate_policy_gradient(
                    action=a,
                    reward=r,
                    pscore=p,
                    q_x_a_hat=q_x_a_hat_,
                    pi_0=pi_0_,
                    pi=pi,
                ).mean()
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
            self.train_loss.append(loss_epoch)
        pi_train = self.predict(D_H)
        self.train_value.append((q_x_a_train * pi_train).sum(1).mean())
        pi_test = self.predict(D_test)
        self.test_value.append((q_x_a_test * pi_test).sum(1).mean())

    def _create_train_data_for_opl(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: np.ndarray,
        q_x_a_hat: np.ndarray,
        pi_0: np.ndarray,
        **kwargs,
    ) -> tuple:
        dataset = NNPolicyDataset(
            torch.from_numpy(context).float(),
            torch.from_numpy(action).long(),
            torch.from_numpy(reward).float(),
            torch.from_numpy(pscore).float(),
            torch.from_numpy(q_x_a_hat).float(),
            torch.from_numpy(pi_0).float(),
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
        )

        return data_loader

    def _estimate_policy_gradient(
        self,
        action: torch.Tensor,
        reward: torch.Tensor,
        pscore: torch.Tensor,
        q_x_a_hat: torch.Tensor,
        pi: torch.Tensor,
        pi_0: torch.Tensor,
    ) -> torch.Tensor:
        current_pi = pi.detach()
        log_prob = torch.log(pi + self.log_eps)
        idx_tensor = torch.arange(action.shape[0], dtype=torch.long)

        q_x_a_hat_factual = q_x_a_hat[idx_tensor, action]
        iw = current_pi[idx_tensor, action] / pscore
        estimated_policy_grad_arr = iw * (reward - q_x_a_hat_factual)
        estimated_policy_grad_arr *= log_prob[idx_tensor, action]
        estimated_policy_grad_arr += torch.sum(q_x_a_hat * current_pi * log_prob, dim=1)

        return estimated_policy_grad_arr

    def predict(self, D_test: np.ndarray) -> np.ndarray:

        self.nn_model.eval()
        x = torch.from_numpy(D_test["x"]).float()
        return self.nn_model(x).detach().numpy()


        
@dataclass
class LongTermCIBasedOPL(TypicalOPL):
    """An OPL method based on the long-term CI framework under the surrogacy assumption."""

    g_x_s_model = MLPRegressor(
        hidden_layer_sizes=(30, 30, 30), random_state=12345
    )

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()

    def fit(self, D_H: dict, D_E: dict, D_test: dict) -> None:
        x, a, pscore, pi_0  = D_E["x"], D_E["actions"], D_E["pscore"], D_E["pi_0"]
        self.g_x_s_model.fit(D_H["x_s"], D_H["r"])
        g_x_s_hat = self.g_x_s_model.predict(D_E["x_s"])
        q_x_a_hat = np.zeros((a.shape[0], self.n_actions))

        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.AdamW(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")

        training_data_loader = self._create_train_data_for_opl(
            x, a, g_x_s_hat, pscore, q_x_a_hat, pi_0
        )

        # start policy training
        x_test = D_test["x"]
        q_x_a_train, q_x_a_test = D_H["q_x_a"], D_test["q_x_a"]
        for _ in range(self.max_iter):
            pi_train = self.predict(D_H)
            self.train_value.append((q_x_a_train * pi_train).sum(1).mean())
            pi_test = self.predict(D_test)
            self.test_value.append((q_x_a_test * pi_test).sum(1).mean())

            loss_epoch = 0.0
            self.nn_model.train()
            for x, a, r, p, q_x_a_hat_, pi_0_ in training_data_loader:
                optimizer.zero_grad()
                pi = self.nn_model(x)
                loss = -self._estimate_policy_gradient(
                    action=a,
                    reward=r,
                    pscore=p,
                    q_x_a_hat=q_x_a_hat_,
                    pi_0=pi_0_,
                    pi=pi,
                ).mean()
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
            self.train_loss.append(loss_epoch)
        pi_train = self.predict(D_H)
        self.train_value.append((q_x_a_train * pi_train).sum(1).mean())
        pi_test = self.predict(D_test)
        self.test_value.append((q_x_a_test * pi_test).sum(1).mean())

    def _estimate_policy_gradient(
        self,
        action: torch.Tensor,
        reward: torch.Tensor,
        pscore: torch.Tensor,
        q_x_a_hat: torch.Tensor,
        pi: torch.Tensor,
        pi_0: torch.Tensor,
    ) -> torch.Tensor:
        current_pi = pi.detach()
        log_prob = torch.log(pi + self.log_eps)
        idx_tensor = torch.arange(action.shape[0], dtype=torch.long)

        q_x_a_hat_factual = q_x_a_hat[idx_tensor, action]
        iw = current_pi[idx_tensor, action] / pscore
        estimated_policy_grad_arr = iw * (reward - q_x_a_hat_factual)
        estimated_policy_grad_arr *= log_prob[idx_tensor, action]
        estimated_policy_grad_arr += torch.sum(q_x_a_hat * current_pi * log_prob, dim=1)

        return estimated_policy_grad_arr
    
    

@dataclass
class LongTermOPL(TypicalOPL):
    """Our proposed method using both short-term and long-term rewards."""

    pi_a_x_s_model = MLPClassifier(
        hidden_layer_sizes=(30, 30, 30), random_state=12345
    )

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__post_init__()

    def fit(self, D_H: dict, D_E: dict, D_test: dict) -> None:
        x, a, r, pi_0  = D_H["x"], D_H["actions"], D_H["r"], D_H["pi_0"]
        q_x_a_hat = estimate_q_x_a_via_regression(D_H)

        x_s_ = np.concatenate([D_H["x_s"], D_E["x_s"]])
        actions_ = np.concatenate([a, D_E["actions"]])
        self.pi_a_x_s_model.fit(x_s_, actions_)
        pi_a_x_s_hat = self.pi_a_x_s_model.predict_proba(D_H["x_s"])

        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.AdamW(
                self.nn_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")

        training_data_loader = self._create_train_data_for_opl(
            x, a, r, q_x_a_hat, pi_0, pi_a_x_s_hat
        )

        # start policy training
        x_test = D_test["x"]
        q_x_a_train, q_x_a_test = D_H["q_x_a"], D_test["q_x_a"]
        for _ in range(self.max_iter):
            pi_train = self.predict(D_H)
            self.train_value.append((q_x_a_train * pi_train).sum(1).mean())
            pi_test = self.predict(D_test)
            self.test_value.append((q_x_a_test * pi_test).sum(1).mean())

            loss_epoch = 0.0
            self.nn_model.train()
            for x, a, r, q_x_a_hat_, pi_0_, pi_a_x_s_hat_ in training_data_loader:
                optimizer.zero_grad()
                pi = self.nn_model(x)
                loss = -self._estimate_policy_gradient(
                    action=a,
                    reward=r,
                    q_x_a_hat=q_x_a_hat_,
                    pi_0=pi_0_,
                    pi_a_x_s_hat=pi_a_x_s_hat_,
                    pi=pi,
                ).mean()
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
            self.train_loss.append(loss_epoch)
        pi_train = self.predict(D_H)
        self.train_value.append((q_x_a_test * pi_test).sum(1).mean())
        pi_test = self.predict(D_test)
        self.test_value.append((q_x_a_test * pi_test).sum(1).mean())

    def _estimate_policy_gradient(
        self,
        action: torch.Tensor,
        reward: torch.Tensor,
        q_x_a_hat: torch.Tensor,
        pi: torch.Tensor,
        pi_0: torch.Tensor,
        pi_a_x_s_hat: torch.Tensor,
    ) -> torch.Tensor:
        current_pi = pi.detach()
        log_prob = torch.log(pi + self.log_eps)
        idx_tensor = torch.arange(action.shape[0], dtype=torch.long)

        q_x_a_hat_factual = q_x_a_hat[idx_tensor, action]
        iw = ((current_pi / pi_0) * pi_a_x_s_hat).sum(1)
        estimated_policy_grad_arr = iw * (reward - q_x_a_hat_factual)
        estimated_policy_grad_arr *= log_prob[idx_tensor, action]
        estimated_policy_grad_arr += torch.sum(q_x_a_hat * current_pi * log_prob, dim=1)

        return estimated_policy_grad_arr

    
    
def run_all_policy_learning_methods(
    D_H: dict, D_E_0: dict, D_test: dict, seed: int = 12345
) -> dict:
    """Run all methods to learn a new policy to optimize the long-term expected reward simultaneously."""
    torch.manual_seed(seed)
    true_value_of_learned_policies = dict()
    x_dim, n_actions = D_H["x"].shape[1], D_H["n_actions"]
        
    reg_based = RegBasedPolicyLearner()
    reg_based.fit(D_H)
    pi_reg_based = reg_based.predict(D_test)
    true_value_of_learned_policies["reg_based"] =\
        (D_test["q_x_a"] * pi_reg_based).sum(1).mean()
    
    typical_opl = TypicalOPL(x_dim=x_dim, n_actions=n_actions, off_policy_objective="ips")
    typical_opl.fit(D_H, D_test)
    pi_typical_opl = typical_opl.predict(D_test)
    true_value_of_learned_policies["ips-pg"] =\
        (D_test["q_x_a"] * pi_typical_opl).sum(1).mean()
    
    typical_opl = TypicalOPL(x_dim=x_dim, n_actions=n_actions, off_policy_objective="dr")
    typical_opl.fit(D_H, D_test)
    pi_typical_opl = typical_opl.predict(D_test)
    true_value_of_learned_policies["dr-pg"] =\
        (D_test["q_x_a"] * pi_typical_opl).sum(1).mean()
    
    long_term_opl = LongTermOPL(x_dim=x_dim, n_actions=n_actions)
    long_term_opl.fit(D_H, D_E_0, D_test)
    pi_long_term_opl = long_term_opl.predict(D_test)
    true_value_of_learned_policies["long_term_opl"] =\
        (D_test["q_x_a"] * pi_long_term_opl).sum(1).mean()
    
    return true_value_of_learned_policies