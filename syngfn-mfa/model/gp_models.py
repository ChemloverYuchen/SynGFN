import torch
import numpy as np
import abc

from scipy.stats import spearmanr

from botorch.models import SingleTaskGP, SingleTaskVariationalGP, KroneckerMultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from gpytorch.utils.memoize import clear_cache_hook
from gpytorch import likelihoods, kernels

from model.metrics import quantile_calibration

from gpytorch.variational import IndependentMultitaskVariationalStrategy
from gpytorch.settings import cholesky_jitter
from gpytorch.lazy import ConstantDiagLazyTensor
from gpytorch import lazify
from .botorch_models import (
    SingleTaskMultiFidelityVariationalGP,
    SingleTaskMultiFidelityLikeBotorchVariationalGP,
)
from gpytorch.kernels.kernel import ProductKernel


class BaseGPSurrogate(abc.ABC):
    def __init__(
        self,
        max_shift,
        mask_size,
        gp_lr,
        enc_lr,
        bs,
        num_epochs,
        holdout_ratio,
        early_stopping,
        patience,
        eval_period,
        tokenizer,
        encoder,
        encoder_wd=0.0,
        bootstrap_ratio=None,
        min_num_train=128,
        task_noise_init=0.01,
        lengthscale_init=0.7,
        *args,
        **kwargs,
    ):
        self.gp_lr = gp_lr
        self.enc_lr = enc_lr
        self.bs = bs
        self.encoder_wd = encoder_wd
        self.num_epochs = num_epochs
        self.holdout_ratio = holdout_ratio
        self.early_stopping = early_stopping
        self.patience = patience
        self.eval_period = eval_period
        self.bootstrap_ratio = bootstrap_ratio
        self.min_num_train = min_num_train
        self.task_noise_init = task_noise_init
        self.lengthscale_init = lengthscale_init
        self.tokenizer = tokenizer

        # self._set_transforms(tokenizer, max_shift, mask_size)

    def get_features(self, seq_array, batch_size=None, transform=None):
        original_shape = seq_array.shape[:-1]
        flat_seq_array = seq_array.flatten(end_dim=-2)

        enc_seq_array = seq_array.to(self.device)
        features = self.encoder.get_features(enc_seq_array)

        return features.view(*original_shape, -1)

    def reshape_targets(self, targets):
        return targets

    def evaluate(self, loader, final=False, *args, **kwargs):
        self.eval()
        targets, y_mean, y_std, f_std = [], [], [], []
        # print("\nUser-Defined Warning: Converting states in test loader to integer for surrogate evaluation.")
        with torch.no_grad():
            for (
                input_batch,
                target_batch,
            ) in loader:
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                features = self.get_features(
                    input_batch.to(self.device), transform=None
                )
                f_dist = self(features)
                y_dist = self.likelihood(f_dist)

                target_batch = self.reshape_targets(target_batch)  # torch.Size([3, 45])
                if target_batch.ndim == 0:
                    # create a batch dimension
                    target_batch = target_batch.unsqueeze(0)  # 1
                else:
                    target_batch = target_batch.squeeze(-1)
                targets.append(target_batch.to(features.device).cpu())
                if y_dist.mean.shape == target_batch.shape:  # True
                    f_std.append(f_dist.variance.sqrt().cpu())
                    y_mean.append(y_dist.mean.cpu())
                    y_std.append(y_dist.variance.sqrt().cpu())
                else:
                    f_std.append(f_dist.variance.sqrt().cpu().transpose(-1, -2))
                    y_mean.append(y_dist.mean.cpu().transpose(-1, -2))
                    y_std.append(y_dist.variance.sqrt().cpu().transpose(-1, -2))

        # TODO: figure out why these are getting flipped
        try:
            targets = torch.cat(targets).view(len(loader.dataset), -1)
            cat_dim = 0
        except:
            targets = torch.cat(targets, -1).view(len(loader.dataset), -1)
            cat_dim = -1
        f_std = torch.cat(f_std, cat_dim).view(len(loader.dataset), -1)
        y_mean = torch.cat(y_mean, cat_dim).view(len(loader.dataset), -1)
        y_std = torch.cat(y_std, cat_dim).view(len(loader.dataset), -1)

        assert y_mean.shape == targets.shape

        rmse = (y_mean - targets).pow(2).mean().sqrt()
        nll = -torch.distributions.Normal(y_mean, y_std).log_prob(targets).mean()
        cal_metrics = quantile_calibration(y_mean, y_std, targets)
        ece = cal_metrics["ece"]
        occ_diff = cal_metrics["occ_diff"]

        spearman_rho = 0
        for idx in range(targets.size(-1)):
            spearman_rho += spearmanr(
                targets[..., idx], y_mean[..., idx]
            ).correlation / targets.size(-1)

        metrics = {
            f"test_nll": nll.item(),
            f"test_rmse": rmse.item(),
            f"test_s_rho": spearman_rho,
            f"test_ece": ece,
            f"test_occ_diff": occ_diff,
            f"test_post_var": (f_std**2).mean().item(),
        }

        if final == True:
            return y_std

        if hasattr(self.likelihood, "task_noises"):
            metrics["noise"] = self.likelihood.task_noises.mean().item()
        elif hasattr(self.likelihood, "noise"):
            metrics["noise"] = self.likelihood.noise.mean().item()
        else:
            pass

        covar_module = (
            self.model.covar_module if hasattr(self, "model") else self.covar_module
        )
        if (
            hasattr(covar_module, "base_kernel")
            and hasattr(covar_module.base_kernel, "lengthscale")
            and covar_module.base_kernel.lengthscale is not None
        ):
            metrics["lengthscale"] = covar_module.base_kernel.lengthscale.mean().item()
        elif hasattr(covar_module, "data_covar_module"):
            metrics[
                "lengthscale"
            ] = covar_module.data_covar_module.lengthscale.mean().item()
        elif (
            hasattr(covar_module, "lengthscale")
            and covar_module.lengthscale is not None
        ):
            metrics["lengthscale"] = covar_module.lengthscale.mean().item()
        else:
            pass

        if (
            hasattr(covar_module, "outputscale")
            and covar_module.outputscale is not None
        ):
            metrics["outputscale"] = covar_module.outputscale.mean().item()

        return metrics

    @property
    def param_groups(self):

        gp_hypers = dict(params=[], lr=self.gp_lr)
        noise_group = dict(params=[], lr=self.gp_lr)
        inducing_point_group = dict(params=[], lr=self.gp_lr)
        variational_group = dict(params=[], lr=self.gp_lr)

        for name, param in self.named_parameters():
            if name.split(".")[0] == "encoder":
                continue
            if "noise" in name:
                noise_group["params"].append(param)
            elif "inducing_points" in name:
                inducing_point_group["params"].append(param)
            elif "variational_distribution" in name:
                variational_group["params"].append(param)
            else:
                gp_hypers["params"].append(param)

        param_groups = [gp_hypers]

        if hasattr(self, "encoder") and hasattr(self.encoder, "param_groups"):
            param_groups.extend(self.encoder.param_groups(self.enc_lr, self.encoder_wd))

        if len(noise_group["params"]) > 0:
            param_groups.append(noise_group)

        if len(inducing_point_group["params"]) > 0:
            param_groups.append(inducing_point_group)

        if len(variational_group["params"]) > 0:
            param_groups.append(variational_group)

        return param_groups


class SingleTaskSVGP(BaseGPSurrogate, SingleTaskVariationalGP):
    def __init__(
        self,
        feature_dim,
        out_dim,
        num_inducing_points,
        encoder,
        device,
        float_precision,
        noise_constraint=None,
        lengthscale_prior=None,
        outcome_transform=None,
        input_transform=None,
        learn_inducing_points=True,
        mll_beta=1.0,
        kernel="matern",
        noise_prior=None,
        *args,
        **kwargs,
    ):
        self.device = device
        self.float = float_precision
        BaseGPSurrogate.__init__(self, encoder=encoder, *args, **kwargs)
        self.num_inducing_points = num_inducing_points  # 64

        if out_dim == 1:
            if kernel == "matern":
                covar_module = kernels.MaternKernel(
                    ard_num_dims=feature_dim, lengthscale_prior=lengthscale_prior
                )
            elif kernel == "rbf":
                covar_module = kernels.RBFKernel(
                    ard_num_dims=feature_dim, lengthscale_prior=lengthscale_prior
                )
            covar_module.initialize(lengthscale=self.lengthscale_init)
            likelihood = likelihoods.GaussianLikelihood(
                noise_constraint=noise_constraint, noise_prior=noise_prior
            )
            likelihood.initialize(noise=self.task_noise_init)
        else:
            raise NotImplementedError(
                "More than one output dim not supported. Refer to lambo repo if you really wanna"
            )

        # initialize GP
        dummy_X = 2 * (
            torch.rand(num_inducing_points, feature_dim).to(self.device, self.float)
            - 0.5
        )
        dummy_Y = torch.randn(num_inducing_points, out_dim).to(self.device, self.float)
        covar_module = (
            covar_module
            if covar_module is None
            else covar_module.to(self.device, self.float)
        )

        self.base_cls = SingleTaskVariationalGP
        self.base_cls.__init__(
            self,
            dummy_X,
            dummy_Y,
            likelihood,
            out_dim,
            learn_inducing_points,
            covar_module=covar_module,
            inducing_points=dummy_X,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self.encoder = encoder.to(self.device, self.float)
        self.mll_beta = mll_beta  # 0.01

    def clear_cache(self):
        clear_cache_hook(self)
        clear_cache_hook(self.model)
        clear_cache_hook(self.model.variational_strategy)
        if hasattr(
            self.model.variational_strategy, "base_variational_strategy"
        ):  # True
            clear_cache_hook(self.model.variational_strategy.base_variational_strategy)

    def forward(self, inputs):
        assert isinstance(inputs, torch.Tensor)
        features = (
            self.get_features(inputs, self.bs)
            if isinstance(inputs, np.ndarray)
            else inputs
        )
        res = self.base_cls.forward(self, features)
        return res

    def posterior(self, X, output_indices=None, observation_noise=False, **kwargs):
        self.clear_cache()
        # features = self.get_features(X.to(torch.long))
        # if isinstance(X, np.ndarray)
        # else X
        return self.base_cls.posterior(
            self, X, output_indices, observation_noise, **kwargs
        )

    def set_train_data(self, inputs=None, targets=None, strict=True):
        self.clear_cache()

    def reshape_targets(self, targets):
        if targets.shape[-1] > 1:
            return targets
        else:
            return targets.squeeze(-1)

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def batch_shape(self):
        """
        This is a batch shape from an I/O perspective. For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        to the `posterior` method returns a Posterior object over an output of
        shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.
        """
        return torch.Size([])

    def initialize_var_dist_sgpr(self, train_x, train_y, noise_lb):
        """
        This is only intended for whitened variational distributions and gaussian likelihoods
        at present.

        \bar m = L^{-1} m
        \bar S = L^{-1} S L^{-T}

        where $LL^T = K_{uu}$.

        Thus, the optimal \bar m, \bar S are given by
        \bar S = L^T (K_{uu} + \sigma^{-2} K_{uv} K_{vu})^{-1} L
        \bar m = \bar S L^{-1} (K_{uv} y \sigma^{-2})
        """

        if isinstance(
            self.model.variational_strategy, IndependentMultitaskVariationalStrategy
        ):  # True in lambo code but False for me
            ind_pts = (
                self.model.variational_strategy.base_variational_strategy.inducing_points
            )  # Parameter containing: torch.Size([64, 16])
            train_y = train_y.transpose(-1, -2).unsqueeze(-1)
            is_batch_model = True
        else:
            ind_pts = self.model.variational_strategy.inducing_points
            is_batch_model = False

        with cholesky_jitter(1e-4):
            kuu = self.model.covar_module(ind_pts).double()
            kuu_chol = kuu.cholesky()
            kuv = self.model.covar_module(ind_pts, train_x).double()

            if hasattr(self.likelihood, "noise"):
                noise = self.likelihood.noise
            elif hasattr(self.likelihood, "task_noises"):
                noise = self.likelihood.task_noises.view(-1, 1, 1)
            else:
                raise AttributeError
            noise = noise.clamp(min=noise_lb).double()

            if len(train_y.shape) < len(kuv.shape):
                train_y = train_y.unsqueeze(-1)
            if len(noise.shape) < len(kuv.shape):
                noise = noise.unsqueeze(-1)

            data_term = kuv.matmul(train_y.double()) / noise
            if is_batch_model:
                noise_as_lt = ConstantDiagLazyTensor(
                    noise.squeeze(-1), diag_shape=kuv.shape[-1]
                )
                inner_prod = kuv.matmul(noise_as_lt).matmul(kuv.transpose(-1, -2))
                inner_term = inner_prod + kuu
            else:
                inner_term = kuv @ kuv.transpose(-1, -2) / noise + kuu

            s_mat = kuu_chol.transpose(-1, -2).matmul(
                inner_term.inv_matmul(kuu_chol.evaluate())
            )
            s_root = lazify(s_mat).cholesky().evaluate()
            # the expression below is less efficient but probably more stable
            mean_param = kuu_chol.transpose(-1, -2).matmul(
                inner_term.inv_matmul(data_term)
            )

        mean_param = mean_param.to(train_y)
        s_root = s_root.to(train_y)

        if not is_batch_model:
            self.model.variational_strategy._variational_distribution.variational_mean.data = (
                mean_param.data.detach().squeeze()
            )
            self.model.variational_strategy._variational_distribution.chol_variational_covar.data = (
                s_root.data.detach()
            )
            self.model.variational_strategy.variational_params_initialized.fill_(1)
        else:
            self.model.variational_strategy.base_variational_strategy._variational_distribution.variational_mean.data = (
                mean_param.data.detach().squeeze()
            )
            self.model.variational_strategy.base_variational_strategy._variational_distribution.chol_variational_covar.data = (
                s_root.data.detach()
            )
            self.model.variational_strategy.base_variational_strategy.variational_params_initialized.fill_(
                1
            )

##
class SingleTaskMultiFidelityLikeBotorchSVGP(
    BaseGPSurrogate, SingleTaskMultiFidelityLikeBotorchVariationalGP
):
    def __init__(
        self,
        feature_dim,
        out_dim,
        num_inducing_points,
        encoder,
        device,
        float_precision,
        n_fid,
        is_fid_param_nn,
        noise_constraint=None,
        lengthscale_prior=None,
        outcome_transform=None,
        input_transform=None,
        learn_inducing_points=True,
        mll_beta=1.0,
        kernel="matern",
        noise_prior=None,
        *args,
        **kwargs,
    ):
        self.device = device
        self.float = float_precision
        BaseGPSurrogate.__init__(self, encoder=encoder, *args, **kwargs)
        self.num_inducing_points = num_inducing_points  # 64
        self.is_fid_param_nn = is_fid_param_nn
        if out_dim == 1:
            likelihood = likelihoods.GaussianLikelihood(
                noise_constraint=noise_constraint, noise_prior=noise_prior
            )
            likelihood.initialize(noise=self.task_noise_init)

            likelihood.initialize(noise=self.task_noise_init)
        else:
            raise NotImplementedError(
                "More than one output dim not supported. Refer to lambo repo if you really wanna"
            )

        # initialize GP
        # +1 for the fidelity dimension
        dummy_X = 2 * (
            torch.rand(num_inducing_points, feature_dim).to(self.device, self.float)
            - 0.5
        )
        dummy_fid = torch.randint(0, n_fid, (num_inducing_points, 1)).to(
            self.device, self.float
        )
        dummy_X = torch.cat([dummy_X, dummy_fid], dim=1)
        dummy_Y = torch.randn(num_inducing_points, out_dim).to(self.device, self.float)

        self.base_cls = SingleTaskMultiFidelityLikeBotorchVariationalGP

        self.base_cls.__init__(
            self,
            dummy_X,
            dummy_Y,
            likelihood,
            out_dim,
            learn_inducing_points,
            covar_module=None,
            inducing_points=dummy_X,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self.encoder = encoder.to(self.device, self.float)
        self.mll_beta = mll_beta

    def clear_cache(self):
        clear_cache_hook(self)
        clear_cache_hook(self.model)
        clear_cache_hook(self.model.variational_strategy)
        if hasattr(self.model.variational_strategy, "base_variational_strategy"):
            clear_cache_hook(self.model.variational_strategy.base_variational_strategy)

    def get_features(self, seq_array, batch_size=None, transform=None):
        original_shape = seq_array.shape[:-1]
        flat_seq_array = seq_array.flatten(end_dim=-2)
        enc_seq_array = seq_array.to(self.device)
        state_array = enc_seq_array[..., :-1]
        fid_array = seq_array[..., -1].to(self.device)
        if self.is_fid_param_nn == True:
            features = self.encoder.get_features(enc_seq_array)
        else:
            features = self.encoder.get_features(state_array)

        features = torch.cat([features, fid_array.unsqueeze(-1)], dim=-1)
        return features.view(*original_shape, -1)

    def forward(self, features):
        assert isinstance(features, torch.Tensor)
        res = self.base_cls.forward(self, features)
        return res

    def posterior(self, X, output_indices=None, observation_noise=False, **kwargs):
        self.clear_cache()
        return self.base_cls.posterior(
            self, X, output_indices, observation_noise, **kwargs
        )

    def set_train_data(self, inputs=None, targets=None, strict=True):
        self.clear_cache()

    def reshape_targets(self, targets):
        if targets.shape[-1] > 1:
            return targets
        else:
            return targets.squeeze(-1)

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def batch_shape(self):
        """
        This is a batch shape from an I/O perspective. For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        to the `posterior` method returns a Posterior object over an output of
        shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.
        """
        return torch.Size([])

    def initialize_var_dist_sgpr(self, train_x, train_y, noise_lb):
        """
        This is only intended for whitened variational distributions and gaussian likelihoods
        at present.

        \bar m = L^{-1} m
        \bar S = L^{-1} S L^{-T}

        where $LL^T = K_{uu}$.

        Thus, the optimal \bar m, \bar S are given by
        \bar S = L^T (K_{uu} + \sigma^{-2} K_{uv} K_{vu})^{-1} L
        \bar m = \bar S L^{-1} (K_{uv} y \sigma^{-2})
        """

        if isinstance(
            self.model.variational_strategy, IndependentMultitaskVariationalStrategy
        ):  # True in lambo code but False for me
            ind_pts = (
                self.model.variational_strategy.base_variational_strategy.inducing_points
            )  # Parameter containing: torch.Size([64, 16])
            train_y = train_y.transpose(-1, -2).unsqueeze(-1)
            is_batch_model = True
        else:
            ind_pts = self.model.variational_strategy.inducing_points
            is_batch_model = False

        with cholesky_jitter(1e-4):
            kuu = self.model.covar_module(ind_pts).double()
            kuu_chol = kuu.cholesky()
            kuv = self.model.covar_module(ind_pts, train_x).double()

            if hasattr(self.likelihood, "noise"):  # False
                noise = self.likelihood.noise
            elif hasattr(self.likelihood, "task_noises"):
                noise = self.likelihood.task_noises.view(-1, 1, 1)
            else:
                raise AttributeError
            noise = noise.clamp(min=noise_lb).double()  # torch.Size([3, 1, 1])

            if len(train_y.shape) < len(kuv.shape):
                train_y = train_y.unsqueeze(-1)
            if len(noise.shape) < len(kuv.shape):
                noise = noise.unsqueeze(-1)

            data_term = kuv.matmul(train_y.double()) / noise  # torch.Size([3, 64, 1])
            if is_batch_model:  # True
                noise_as_lt = ConstantDiagLazyTensor(
                    noise.squeeze(-1), diag_shape=kuv.shape[-1]
                )
                inner_prod = kuv.matmul(noise_as_lt).matmul(kuv.transpose(-1, -2))
                inner_term = inner_prod + kuu
            else:
                inner_term = kuv @ kuv.transpose(-1, -2) / noise + kuu

            s_mat = kuu_chol.transpose(-1, -2).matmul(
                inner_term.inv_matmul(kuu_chol.evaluate())
            )  # torch.Size([3, 64, 64])
            s_root = lazify(s_mat).cholesky().evaluate()  # torch.Size([3, 64, 64])
            # the expression below is less efficient but probably more stable
            mean_param = kuu_chol.transpose(-1, -2).matmul(
                inner_term.inv_matmul(data_term)
            )

        mean_param = mean_param.to(train_y)  # torch.Size([3, 64, 1])
        s_root = s_root.to(train_y)  # torch.Size([3, 64, 64])

        if not is_batch_model:  # True
            self.model.variational_strategy._variational_distribution.variational_mean.data = (
                mean_param.data.detach().squeeze()
            )
            self.model.variational_strategy._variational_distribution.chol_variational_covar.data = (
                s_root.data.detach()
            )
            self.model.variational_strategy.variational_params_initialized.fill_(1)
        else:
            self.model.variational_strategy.base_variational_strategy._variational_distribution.variational_mean.data = (
                mean_param.data.detach().squeeze()
            )
            self.model.variational_strategy.base_variational_strategy._variational_distribution.chol_variational_covar.data = (
                s_root.data.detach()
            )
            self.model.variational_strategy.base_variational_strategy.variational_params_initialized.fill_(
                1
            )


# TODO: Adopt inheritance
class SingleTaskMultiFidelitySVGP(
    BaseGPSurrogate, SingleTaskMultiFidelityVariationalGP
):
    def __init__(
        self,
        feature_dim,
        out_dim,
        num_inducing_points,
        encoder,
        device,
        float_precision,
        n_fid,
        is_fid_param_nn,
        noise_constraint=None,
        lengthscale_prior=None,
        index_kernel_rank=1,
        outcome_transform=None,
        input_transform=None,
        learn_inducing_points=True,
        mll_beta=1.0,
        kernel="matern",
        noise_prior=None,
        *args,
        **kwargs,
    ):
        self.device = device
        self.float = float_precision
        BaseGPSurrogate.__init__(self, encoder=encoder, *args, **kwargs)
        self.num_inducing_points = num_inducing_points  # 64
        self.is_fid_param_nn = is_fid_param_nn
        active_dims = torch.arange(feature_dim).to(device)
        if out_dim == 1:
            if kernel == "matern":
                covar_module_x = kernels.MaternKernel(
                    ard_num_dims=feature_dim,
                    lengthscale_prior=lengthscale_prior,
                )
            elif kernel == "rbf":
                covar_module_x = kernels.RBFKernel(
                    ard_num_dims=feature_dim, lengthscale_prior=lengthscale_prior
                )
            covar_module_x.initialize(lengthscale=self.lengthscale_init)
            likelihood = likelihoods.GaussianLikelihood(
                noise_constraint=noise_constraint, noise_prior=noise_prior
            )
            likelihood.initialize(noise=self.task_noise_init)
            covar_module_fidelity = kernels.IndexKernel(
                num_tasks=n_fid, rank=index_kernel_rank
            )
            covar_module = ProductKernel(covar_module_x, covar_module_fidelity)

            likelihood.initialize(noise=self.task_noise_init)
        else:
            raise NotImplementedError(
                "More than one output dim not supported. Refer to lambo repo if you really wanna"
            )

        # initialize GP
        # +1 for the fidelity dimension
        dummy_X = 2 * (
            torch.rand(num_inducing_points, feature_dim).to(self.device, self.float)
            - 0.5
        )
        dummy_fid = torch.randint(0, n_fid, (num_inducing_points, 1)).to(
            self.device, self.float
        )
        dummy_X = torch.cat([dummy_X, dummy_fid], dim=1)
        dummy_Y = torch.randn(num_inducing_points, out_dim).to(self.device, self.float)

        covar_module_x = (
            covar_module_x
            if covar_module_x is None
            else covar_module_x.to(self.device, self.float)
        )
        covar_module_fidelity = (
            covar_module_fidelity
            if covar_module_fidelity is None
            else covar_module_fidelity.to(self.device, self.float)
        )

        self.base_cls = SingleTaskMultiFidelityVariationalGP
        self.base_cls.__init__(
            self,
            dummy_X,
            dummy_Y,
            likelihood,
            out_dim,
            learn_inducing_points,
            covar_module_x=covar_module_x,
            covar_module_fidelity=covar_module_fidelity,
            inducing_points=dummy_X,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self.encoder = encoder.to(self.device, self.float)
        self.mll_beta = mll_beta

    def clear_cache(self):
        clear_cache_hook(self)
        clear_cache_hook(self.model)
        clear_cache_hook(self.model.variational_strategy)
        if hasattr(
            self.model.variational_strategy, "base_variational_strategy"
        ):  # True
            clear_cache_hook(self.model.variational_strategy.base_variational_strategy)

    def get_features(
        self, seq_array, batch_size=None, transform=None
    ):  # batch_size not used
        original_shape = seq_array.shape[:-1]
        flat_seq_array = seq_array.flatten(end_dim=-2)
        enc_seq_array = seq_array.to(self.device)
        state_array = enc_seq_array[..., :-1]
        fid_array = seq_array[..., -1].to(self.device)
        if self.is_fid_param_nn == True:
            features = self.encoder.get_features(enc_seq_array)
        else:
            features = self.encoder.get_features(state_array)

        features = torch.cat([features, fid_array.unsqueeze(-1)], dim=-1)
        return features.view(*original_shape, -1)

    def forward(self, features):
        assert isinstance(features, torch.Tensor)
        res = self.base_cls.forward(self, features)
        return res

    def posterior(self, X, output_indices=None, observation_noise=False, **kwargs):
        self.clear_cache()
        return self.base_cls.posterior(
            self, X, output_indices, observation_noise, **kwargs
        )

    def set_train_data(self, inputs=None, targets=None, strict=True):
        self.clear_cache()

    def reshape_targets(self, targets):
        if targets.shape[-1] > 1:
            return targets
        else:
            return targets.squeeze(-1)

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def batch_shape(self):

        # This is a batch shape from an I/O perspective. For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        # to the `posterior` method returns a Posterior object over an output of
        # shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.

        return torch.Size([])

    def evaluate(self, loader, *args, **kwargs):
        self.eval()
        targets, y_mean, y_std, f_std = [], [], [], []
        with torch.no_grad():
            for (
                input_batch,
                target_batch,
            ) in loader:

                features = self.get_features(
                    input_batch.to(self.device), transform=None
                )
                f_dist = self(features)
                y_dist = self.likelihood(f_dist)

                target_batch = self.reshape_targets(target_batch)
                if target_batch.ndim == 0:
                    # create a batch dimension
                    target_batch = target_batch.unsqueeze(0)
                else:
                    target_batch = target_batch.squeeze(-1)
                targets.append(target_batch.to(features.device).cpu())
                if y_dist.mean.shape == target_batch.shape:
                    f_std.append(f_dist.variance.sqrt().cpu())
                    y_mean.append(y_dist.mean.cpu())
                    y_std.append(y_dist.variance.sqrt().cpu())
                else:
                    f_std.append(f_dist.variance.sqrt().cpu().transpose(-1, -2))
                    y_mean.append(y_dist.mean.cpu().transpose(-1, -2))
                    y_std.append(y_dist.variance.sqrt().cpu().transpose(-1, -2))

        try:
            targets = torch.cat(targets).view(len(loader.dataset), -1)
            cat_dim = 0
        except:
            targets = torch.cat(targets, -1).view(len(loader.dataset), -1)
            cat_dim = -1

        f_std = torch.cat(f_std, cat_dim).view(len(loader.dataset), -1)
        y_mean = torch.cat(y_mean, cat_dim).view(len(loader.dataset), -1)
        y_std = torch.cat(y_std, cat_dim).view(len(loader.dataset), -1)

        assert y_mean.shape == targets.shape

        rmse = (y_mean - targets).pow(2).mean().sqrt()
        nll = -torch.distributions.Normal(y_mean, y_std).log_prob(targets).mean()
        cal_metrics = quantile_calibration(y_mean, y_std, targets)
        ece = cal_metrics["ece"]
        occ_diff = cal_metrics["occ_diff"]

        spearman_rho = 0
        for idx in range(targets.size(-1)):
            spearman_rho += spearmanr(
                targets[..., idx], y_mean[..., idx]
            ).correlation / targets.size(-1)

        metrics = {
            f"test_nll": nll.item(),
            f"test_rmse": rmse.item(),
            f"test_s_rho": spearman_rho,
            f"test_ece": ece,
            f"test_occ_diff": occ_diff,
            f"test_post_var": (f_std**2).mean().item(),
        }

        if hasattr(self.likelihood, "task_noises"):
            metrics["noise"] = self.likelihood.task_noises.mean().item()
        elif hasattr(self.likelihood, "noise"):
            metrics["noise"] = self.likelihood.noise.mean().item()
        else:
            pass

        covar_module = (
            self.model.covar_module_x if hasattr(self, "model") else self.covar_module
        )
        if hasattr(covar_module, "base_kernel") and hasattr(
            covar_module.base_kernel, "lengthscale"
        ):
            metrics["lengthscale"] = covar_module.base_kernel.lengthscale.mean().item()
        elif hasattr(covar_module, "data_covar_module"):
            metrics[
                "lengthscale"
            ] = covar_module.data_covar_module.lengthscale.mean().item()
        elif hasattr(covar_module, "lengthscale"):
            metrics["lengthscale"] = covar_module.lengthscale.mean().item()
        else:
            pass

        if hasattr(covar_module, "outputscale"):
            metrics["outputscale"] = covar_module.outputscale.mean().item()

        return metrics

    def initialize_var_dist_sgpr(self, train_x, train_y, noise_lb):

        # This is only intended for whitened variational distributions and gaussian likelihoods
        # at present.

        # \bar m = L^{-1} m
        # \bar S = L^{-1} S L^{-T}

        # where $LL^T = K_{uu}$.

        # Thus, the optimal \bar m, \bar S are given by
        # \bar S = L^T (K_{uu} + \sigma^{-2} K_{uv} K_{vu})^{-1} L
        # \bar m = \bar S L^{-1} (K_{uv} y \sigma^{-2})

        if isinstance(
            self.model.variational_strategy, IndependentMultitaskVariationalStrategy
        ):  # True in lambo code but False for me
            ind_pts = (
                self.model.variational_strategy.base_variational_strategy.inducing_points
            )  # Parameter containing: torch.Size([64, 16])
            train_y = train_y.transpose(-1, -2).unsqueeze(-1)
            is_batch_model = True
        else:
            ind_pts = self.model.variational_strategy.inducing_points
            is_batch_model = False

        with cholesky_jitter(1e-4):
            kuu_x = self.model.covar_module_x(ind_pts[..., :-1]).double()
            kuu_fidelity = self.model.covar_module_fidelity(ind_pts[..., -1:]).double()
            kuu = kuu_x.mul(kuu_fidelity)
            kuu_chol = kuu.cholesky()
            kuv_x = self.model.covar_module_x(
                ind_pts[..., :-1], train_x[..., :-1]
            ).double()
            kuv_fidelity = self.model.covar_module_fidelity(
                ind_pts[..., -1:], train_x[..., -1:]
            ).double()
            kuv = kuv_x.mul(kuv_fidelity)

            if hasattr(self.likelihood, "noise"):  # False
                noise = self.likelihood.noise
            elif hasattr(self.likelihood, "task_noises"):
                noise = self.likelihood.task_noises.view(-1, 1, 1)
            else:
                raise AttributeError
            noise = noise.clamp(min=noise_lb).double()  # torch.Size([3, 1, 1])

            if len(train_y.shape) < len(kuv.shape):
                train_y = train_y.unsqueeze(-1)
            if len(noise.shape) < len(kuv.shape):
                noise = noise.unsqueeze(-1)

            data_term = kuv.matmul(train_y.double()) / noise  # torch.Size([3, 64, 1])
            if is_batch_model:  # True
                noise_as_lt = ConstantDiagLazyTensor(
                    noise.squeeze(-1), diag_shape=kuv.shape[-1]
                )
                inner_prod = kuv.matmul(noise_as_lt).matmul(kuv.transpose(-1, -2))
                inner_term = inner_prod + kuu
            else:
                inner_term = kuv @ kuv.transpose(-1, -2) / noise + kuu

            s_mat = kuu_chol.transpose(-1, -2).matmul(
                inner_term.inv_matmul(kuu_chol.evaluate())
            )
            # TODO: Replace gpytorch.lazy.lazify with linear_operator.to_linear_operato
            s_root = lazify(s_mat).cholesky().evaluate()
            # the expression below is less efficient but probably more stable
            mean_param = kuu_chol.transpose(-1, -2).matmul(
                inner_term.inv_matmul(data_term)
            )

        mean_param = mean_param.to(train_y)
        s_root = s_root.to(train_y)

        if not is_batch_model:
            self.model.variational_strategy._variational_distribution.variational_mean.data = (
                mean_param.data.detach().squeeze()
            )
            self.model.variational_strategy._variational_distribution.chol_variational_covar.data = (
                s_root.data.detach()
            )
            self.model.variational_strategy.variational_params_initialized.fill_(1)
        else:
            self.model.variational_strategy.base_variational_strategy._variational_distribution.variational_mean.data = (
                mean_param.data.detach().squeeze()
            )
            self.model.variational_strategy.base_variational_strategy._variational_distribution.chol_variational_covar.data = (
                s_root.data.detach()
            )
            self.model.variational_strategy.base_variational_strategy.variational_params_initialized.fill_(
                1
            )
