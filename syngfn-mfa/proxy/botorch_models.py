import torch
from botorch.models.model import Model
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.acquisition.cost_aware import CostAwareUtility
from gflownet.gflownet.utils.common import set_device, set_float_precision


class FidelityCostModel(CostAwareUtility):
    def __init__(self, fidelity_weights, fixed_cost, device, float_precision):
        super().__init__()
        self.fidelity_weights = fidelity_weights
        self.device = device
        self.fixed_cost = torch.tensor([fixed_cost], dtype=float_precision).to(
            self.device
        )
        self.float = float_precision

    def forward(self, X, deltas, **kwargs):
        fidelity = X[:, 0, -1]
        scaled_deltas = torch.zeros(X.shape[0], dtype=self.float).to(self.device)
        for fid in self.fidelity_weights.keys():
            idx_fid = torch.where(fidelity == fid)[0]
            cost_fid = torch.tensor([self.fidelity_weights[fid]], dtype=self.float).to(
                self.device
            )
            scaled_deltas[idx_fid] = (deltas[0, idx_fid] / cost_fid) + self.fixed_cost
        scaled_deltas = scaled_deltas.unsqueeze(0)
        return scaled_deltas


class ProxyBotorchUCB(Model):
    def __init__(self, regressor, num_dropout_samples):
        super().__init__()
        self.regressor = regressor
        self._num_outputs = 1
        self.num_dropout_samples = num_dropout_samples

    def posterior(self, X, observation_noise=False, posterior_transform=None):
        super().posterior(X, observation_noise, posterior_transform)

        self.regressor.model.train()
        dim = X.ndim
        if dim == 3:
            X = X.squeeze(-2)

        with torch.no_grad():
            outputs = self.regressor.forward_with_uncertainty(
                X, self.num_dropout_samples
            )
        mean = torch.mean(outputs, dim=1).unsqueeze(-1)
        var = torch.var(outputs, dim=1).unsqueeze(-1)
        # if var is an array of zeros then we add a small value to it
        var = torch.where(var == 0, torch.ones_like(var) * 1e-4, var)

        covar = [torch.diag(var[i]) for i in range(X.shape[0])]
        covar = torch.stack(covar, axis=0)
        mvn = MultivariateNormal(mean, covar)
        posterior = GPyTorchPosterior(mvn)
        return posterior

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def batch_shape(self):
        return torch.Size([])


class ProxyBotorchMES(Model):
    def __init__(self, regressor, num_dropout_samples):
        super().__init__()
        self.regressor = regressor
        self._num_outputs = 1
        self.num_dropout_samples = num_dropout_samples

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def batch_shape(self):
        return torch.Size([])

    def posterior(self, X, observation_noise=False, posterior_transform=None):
        super().posterior(X, observation_noise, posterior_transform)

        self.regressor.model.train()
        dim = X.ndim

        with torch.no_grad():
            outputs = self.regressor.forward_with_uncertainty(
                X, self.num_dropout_samples
            )
        mean = torch.mean(outputs, dim=1).unsqueeze(-1)
        var = torch.var(outputs, dim=1).unsqueeze(-1)
        # if var is an array of zeros then we add a small value to it
        var = torch.where(var == 0, torch.ones_like(var) * 1e-4, var)

        if dim == 2:
            # candidate set
            covar = torch.diag(var)
        elif dim == 4:
            covar = [torch.diag(var[i][0]) for i in range(X.shape[0])]
            covar = torch.stack(covar, axis=0)
            covar = covar.unsqueeze(-1)
        elif dim == 3:
            # max samples
            covar = [torch.diag(var[i]) for i in range(X.shape[0])]
            covar = torch.stack(covar, axis=0)

        mvn = MultivariateNormal(mean, covar)
        posterior = GPyTorchPosterior(mvn)
        return posterior


class MultifidelityOracleModel(Model):
    def __init__(self, oracle, n_fid, device):
        super().__init__()
        self.oracle = oracle
        self._num_outputs = 1
        self.n_fid = n_fid
        self.device = device

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def batch_shape(self):
        return torch.Size([])

    def posterior(self, X, observation_noise=False, posterior_transform=None):
        super().posterior(X, observation_noise, posterior_transform)
        covar_mm = torch.FloatTensor(
            [
                [self.oracle[0].sigma ** 2, (0.015) ** 2, (5e-3) ** 2],
                [(0.015**2), self.oracle[1].sigma ** 2, (4e-3) ** 2],
                [(5e-3) ** 2, (4e-3) ** 2, self.oracle[2].sigma ** 2],
            ]
        )
        if len(X.shape) == 2:
            fid_tensor = X[:, -1]
            state_tensor = X[:, :-1]
            var = torch.zeros(X.shape[0]).to(self.device)
            mean = torch.zeros(X.shape[0]).to(self.device)
            for fid in range(self.n_fid):
                idx_fid = torch.where(fid_tensor == fid)[0]
                var[idx_fid] = self.oracle[fid].sigma ** 2
                mean[idx_fid] = self.oracle[fid](state_tensor[idx_fid])
            # candidate set
            mean = mean * (-0.1)
            covar = torch.diag(var)
        # batch and projected batch
        elif len(X.shape) == 4:
            # combination of target and original fidelity test set
            var_curr_fidelity = torch.zeros(X.shape[0], 1)
            states = X[:, :, 0, :-1].squeeze(-2)
            states = states.squeeze(-2)
            mean_max_fidelity = self.oracle[self.n_fid - 1](states)
            mean_curr_fidelity = torch.ones(mean_max_fidelity.shape).to(self.device)
            fid_tensor = X[:, :, 0, -1]
            var_max_fidelity = self.oracle[self.n_fid - 1].sigma ** 2
            covar = torch.zeros(X.shape[0], 2, 2)
            for fid in range(self.n_fid):
                idx_fid = torch.where(fid_tensor == fid)[0]
                if fid == self.n_fid - 1:
                    var_curr_fidelity = self.oracle[fid].sigma ** 2 + 2e-7
                else:
                    var_curr_fidelity = self.oracle[fid].sigma ** 2
                mean_curr_fidelity[idx_fid] = self.oracle[fid](states[idx_fid])
                covar[idx_fid] = torch.FloatTensor(
                    [
                        [var_curr_fidelity, covar_mm[fid][self.n_fid - 1]],
                        [covar_mm[fid][self.n_fid - 1], var_max_fidelity],
                    ]
                )
            mean_curr_fidelity = mean_curr_fidelity.unsqueeze(-1)
            mean_max_fidelity = mean_max_fidelity.unsqueeze(-1)
            mean = torch.stack((mean_curr_fidelity, mean_max_fidelity), dim=2)
            mean = mean * (-0.1)
            covar = covar.unsqueeze(1).to(self.device)
        # batch
        elif len(X.shape) == 3:
            fid_tensor = X[:, :, -1]
            state_tensor = X[:, :, :-1]
            state_tensor = state_tensor.squeeze(-2)
            mean = torch.zeros((X.shape[0])).to(self.device)
            var = torch.ones((X.shape[0], 1)).to(self.device)
            for fid in range(self.n_fid):
                idx_fid = torch.where(fid_tensor == fid)[0]
                var[idx_fid] = self.oracle[fid].sigma ** 2
                mean[idx_fid] = self.oracle[fid](state_tensor[idx_fid])
            # posterior of max samples
            mean = mean.unsqueeze(-1)
            mean = mean * (-0.1)
            covar = [torch.diag(var[i]) for i in range(X.shape[0])]
            covar = torch.stack(covar, axis=0)

        mean = mean.to(self.device)
        covar = covar.to(self.device)
        mvn = MultivariateNormal(mean, covar)
        posterior = GPyTorchPosterior(mvn)
        return posterior


class MultiFidelityProxyModel(Model):
    """
    Has been tested only with AMP so no mean = mean * (-0.1) over here.
    """

    def __init__(self, regressor, num_dropout_samples, n_fid, device):
        super().__init__()
        self.regressor = regressor
        self._num_outputs = 1
        self.num_dropout_samples = num_dropout_samples
        self.n_fid = n_fid
        self.device = device

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def batch_shape(self):
        return torch.Size([])

    def posterior(self, X, observation_noise=False, posterior_transform=None):
        super().posterior(X, observation_noise, posterior_transform)

        self.regressor.model.train()
        input_dim = X.ndim

        if input_dim == 4:
            X = X.squeeze(1)
            curr_states = X[:, 0, :]
            projected_states = X[:, 1, :]
            with torch.no_grad():
                curr_outputs = self.regressor.forward_with_uncertainty(
                    curr_states, self.num_dropout_samples
                )
                projected_outputs = self.regressor.forward_with_uncertainty(
                    projected_states, self.num_dropout_samples
                )
            outputs = torch.stack((curr_outputs, projected_outputs), dim=1)
            mean = torch.mean(outputs, dim=2)
            var = torch.var(outputs, dim=2)
        else:
            if input_dim == 3:
                X = X.squeeze(1)
            with torch.no_grad():
                outputs = self.regressor.forward_with_uncertainty(
                    X, self.num_dropout_samples
                )
            mean = torch.mean(outputs, dim=1)
            var = torch.var(outputs, dim=1)

        if input_dim == 2:
            covar = torch.diag(var)
        elif input_dim == 4:
            mean = mean.unsqueeze(-2)
            covar = [torch.cov(outputs[i]) for i in range(X.shape[0])]
            covar = torch.stack(covar, axis=0)
            covar = covar.unsqueeze(1)
        elif input_dim == 3:
            mean = mean.unsqueeze(-1)
            var = var.unsqueeze(-1)
            covar = [torch.diag(var[i]) for i in range(X.shape[0])]
            covar = torch.stack(covar, axis=0)

        mvn = MultivariateNormal(mean, covar)
        posterior = GPyTorchPosterior(mvn)
        return posterior
