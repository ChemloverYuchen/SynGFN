import torch
import gpytorch
from tqdm import tqdm
import hydra
from botorch.models.gp_regression_fidelity import (
    SingleTaskMultiFidelityGP,
    SingleTaskGP,
)
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from abc import abstractmethod
from botorch.models.approximate_gp import SingleTaskVariationalGP
from gpytorch.mlls import VariationalELBO
from botorch.settings import debug
from typing import List, Tuple, NewType, Union

class SingleTaskGPRegressor:
    def __init__(self, logger, device, dataset, maximize, env, **kwargs):
        
        self.logger = logger
        self.device = device
        self.float = torch.float32
        
        # Dataset
        self.dataset = dataset# x:smiles列表
        self.n_fid = dataset.n_fid
        self.n_samples = dataset.n_samples

        # Logger
        self.progress = self.logger.progress
        self.target_factor = self.dataset.target_factor

        self.env = env
        
    @abstractmethod
    def init_model(self, train_x, train_y):
        pass

    def fit(self):
        train = self.dataset.train_dataset
        if self.n_fid == 1:
            train_x = train["states"]
            train_x = self.env._statetorch2policy_1(train_x)
            train_x = torch.tensor(train_x, dtype=self.float, device=self.device)
        else:
            train_x = train["states"]
            train_x = self.env.statetorch2policy_1(train_x)
        train_y = train["energies"].unsqueeze(-1)
        self.init_model(train_x, train_y)
        with debug(state=True):
            self.mll = fit_gpytorch_mll(self.mll)

    def get_predictions(self, states, denorm=False):
        detach = True
        if isinstance(states, torch.Tensor) == False:
            states = torch.tensor(states, device=self.device, dtype=self.float)
            detach = False
        if states.ndim == 1:
            states = states.unsqueeze(0)
            
        if self.n_fid == 1:
            states = self.env.statebatch2policy_1(states)
        else:
            states = self.env.statetorch2policy_1(states)
            
        model = self.model
        model.eval()
        model.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = model.posterior(states)
            y_mean = posterior.mean
            y_std = posterior.variance.sqrt()
        if detach == True:
            y_mean = y_mean.detach().cpu().numpy().squeeze(-1)
            y_std = y_std.detach().cpu().numpy().squeeze(-1)
        else:
            y_mean = y_mean.squeeze(-1)
            y_std = y_std.squeeze(-1)
        if denorm == True and self.dataset.normalize_data == True:
            y_mean =torch.tensor(y_mean, dtype=self.float)
            y_mean = (
                y_mean
                * (
                    self.dataset.train_stats["max"].cpu()
                    - self.dataset.train_stats["min"].cpu()
                )
                + self.dataset.train_stats["min"].cpu()
            )
            y_mean = y_mean.squeeze(-1)
        return y_mean, y_std
    
    def __call__(self, states:List[str]):
        scores, _ = self.get_predictions(states, denorm = True)
        return scores
    
    def get_metrics(self, y_mean, y_std, env, states):
        state_oracle_input = states.clone()
        if hasattr(env, "call_oracle_per_fidelity"):
            samples, fidelity = env.statebatch2oracle(state_oracle_input)
            targets = env.call_oracle_per_fidelity(samples, fidelity).detach().cpu()
        elif hasattr(env, "oracle"):
            samples = env.statebatch2oracle(state_oracle_input)
            targets = env.oracle(samples).detach().cpu()
        targets = targets * self.target_factor
        targets_numpy = targets.detach().cpu().numpy()
        targets_numpy = targets_numpy
        rmse = np.sqrt(np.mean((y_mean - targets_numpy) ** 2))
        nll = (
            -torch.distributions.Normal(torch.tensor(y_mean), torch.tensor(y_std))
            .log_prob(targets)
            .mean()
        )
        return rmse, nll

    def plot_predictions(self, states, scores, length, rescale=1):
        n_fid = self.n_fid
        n_states = int(length * length)
        if states.shape[-1] == 3:
            states = states[:, :2]
            states = torch.unique(states, dim=0)
        width = (n_fid) * 5
        fig, axs = plt.subplots(1, n_fid, figsize=(width, 5))
        for fid in range(0, n_fid):
            index = states.long().detach().cpu().numpy()
            grid_scores = np.zeros((length, length))
            grid_scores[index[:, 0], index[:, 1]] = scores[
                fid * n_states : (fid + 1) * n_states
            ]
            if n_fid == 1:
                ax = axs
            else:
                ax = axs[fid]
            if rescale != 1:
                step = int(length / rescale)
            else:
                step = 1
            ax.set_xticks(np.arange(start=0, stop=length, step=step))
            ax.set_yticks(np.arange(start=0, stop=length, step=step))
            ax.imshow(grid_scores)
            if n_fid == 1:
                title = "GP Predictions"
            else:
                title = "GP Predictions with fid {}/{}".format(fid + 1, n_fid)
            ax.set_title(title)
            im = ax.imshow(grid_scores)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.show()
        plt.tight_layout()
        plt.close()
        return fig

    def get_modes(self, states, env):
        num_pick = int((env.length * env.length) / 100) * 5
        percent = num_pick / (env.length * env.length) * 100
        print(
            "\nUser-Defined Warning: Top {}% of states with maximum reward are picked for GP mode metrics".format(
                percent
            )
        )
        state_oracle_input = states.clone()
        if hasattr(env, "call_oracle_per_fidelity"):
            samples, fidelity = env.statebatch2oracle(state_oracle_input)
            targets = env.oracle[-1](samples).detach().cpu()
        elif hasattr(env, "oracle"):
            samples = env.statebatch2oracle(state_oracle_input)
            targets = env.oracle(samples).detach().cpu()
        targets = targets * self.target_factor
        idx_pick = torch.argsort(targets, descending=True)[:num_pick].tolist()
        states_pick = states[idx_pick]
        if hasattr(env, "oracle"):
            self._mode = states_pick
        else:
            fidelities = torch.zeros((len(states_pick) * 3, 1)).to(states_pick.device)
            for i in range(self.n_fid):
                fidelities[i * len(states_pick) : (i + 1) * len(states_pick), 0] = i
            states_pick = states_pick.repeat(self.n_fid, 1)
            state_pick_fid = torch.cat([states_pick, fidelities], dim=1)
            self._mode = state_pick_fid

class MultiFidelitySingleTaskRegressor(SingleTaskGPRegressor):
    def __init__(self, logger, device, dataset, **kwargs):
        super().__init__(logger, device, dataset, **kwargs)

    def init_model(self, train_x, train_y):
        fid_column = train_x.shape[-1] - 1
        self.model = SingleTaskMultiFidelityGP(
            train_x,
            train_y,
            outcome_transform=Standardize(m=1),
            # fid column
            data_fidelity=fid_column,
        )
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )
        self.mll.to(train_x)

class SingleFidelitySingleTaskRegressor(SingleTaskGPRegressor):
    def __init__(self, logger, device, dataset, **kwargs):
        super().__init__(logger, device, dataset, **kwargs)

    def init_model(self, train_x, train_y):
        self.model = SingleTaskGP(
            train_x,
            train_y,
            outcome_transform=Standardize(m=1),
        )
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )
        self.mll.to(train_x)
