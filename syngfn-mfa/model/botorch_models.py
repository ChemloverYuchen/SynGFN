from botorch.models.approximate_gp import (
    ApproximateGP,
    ApproximateGPyTorchModel,
    MIN_INFERRED_NOISE_LEVEL,
)
from typing import Optional, Type, Union
from torch import Tensor
from gpytorch.variational import (
    _VariationalDistribution,
    _VariationalStrategy,
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)
from gpytorch.kernels import Kernel, MaternKernel, IndexKernel, ScaleKernel
from gpytorch.likelihoods import (
    GaussianLikelihood,
    Likelihood,
    MultitaskGaussianLikelihood,
)
from gpytorch.means import ConstantMean, Mean
import copy
import torch
from gpytorch.priors import GammaPrior
from botorch.posteriors.gpytorch import GPyTorchPosterior

from botorch.models.utils.inducing_point_allocators import _pivoted_cholesky_init
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.transforms.input import InputTransform
from botorch.models.utils import validate_input_scaling
from gpytorch.constraints import GreaterThan
from gpytorch.utils.memoize import clear_cache_hook
from gpytorch.module import Module
from gpytorch.distributions import MultivariateNormal
from botorch.models.utils.inducing_point_allocators import (
    GreedyVarianceReduction,
    InducingPointAllocator,
)
from .inducing_point_allocator import MultiFidelityGreedyVarianceReduction
from botorch.models.kernels.linear_truncated_fidelity import (
    LinearTruncatedFidelityKernel,
)


class SingleTaskMultiFidelityVariationalGP(ApproximateGPyTorchModel):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Optional[Tensor] = None,
        likelihood: Optional[Likelihood] = None,
        num_outputs: int = 1,
        learn_inducing_points: bool = True,
        # covar_module: Optional[Kernel] = None,
        covar_module_x: Optional[Kernel] = None,
        covar_module_fidelity: Optional[Kernel] = None,
        mean_module: Optional[Mean] = None,
        variational_distribution: Optional[_VariationalDistribution] = None,
        variational_strategy: Type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Optional[Union[Tensor, int]] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        inducing_point_allocator: Optional[InducingPointAllocator] = None,
    ) -> None:
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if train_Y is not None:
            if outcome_transform is not None:
                train_Y, _ = outcome_transform(train_Y)
            self._validate_tensor_args(X=transformed_X, Y=train_Y)
            validate_input_scaling(train_X=transformed_X, train_Y=train_Y)
            if train_Y.shape[-1] != num_outputs:
                num_outputs = train_Y.shape[-1]

        self._num_outputs = num_outputs
        self._input_batch_shape = train_X.shape[:-2]
        aug_batch_shape = copy.deepcopy(self._input_batch_shape)
        if num_outputs > 1:
            aug_batch_shape += torch.Size([num_outputs])
        self._aug_batch_shape = aug_batch_shape

        if likelihood is None:
            raise NotImplementedError(
                "Likelihood must be provided while initalising user-defined SingleTaskMultiFidelityVariationalGP."
            )
            if num_outputs == 1:
                noise_prior = GammaPrior(1.1, 0.05)
                noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
                likelihood = GaussianLikelihood(
                    noise_prior=noise_prior,
                    batch_shape=self._aug_batch_shape,
                    noise_constraint=GreaterThan(
                        MIN_INFERRED_NOISE_LEVEL,
                        transform=None,
                        initial_value=noise_prior_mode,
                    ),
                )
            else:
                raise NotImplementedError(
                    "Multitask likelihood should not be used in SingleTaskMultiFidelityVariationalGP."
                )
                likelihood = MultitaskGaussianLikelihood(num_tasks=num_outputs)
        else:
            self._is_custom_likelihood = True

        if inducing_point_allocator is None:
            # self._inducing_point_allocator = GreedyVarianceReduction()
            self._inducing_point_allocator = MultiFidelityGreedyVarianceReduction()
        else:
            self._inducing_point_allocator = inducing_point_allocator

        model = _SingleTaskMultiFidelityVariationalGP(
            train_X=transformed_X,
            train_Y=train_Y,
            num_outputs=num_outputs,
            learn_inducing_points=learn_inducing_points,
            # covar_module=covar_module,
            covar_module_x=covar_module_x,
            covar_module_fidelity=covar_module_fidelity,
            mean_module=mean_module,
            variational_distribution=variational_distribution,
            variational_strategy=variational_strategy,
            inducing_points=inducing_points,
            inducing_point_allocator=self._inducing_point_allocator,
        )

        super().__init__(model=model, likelihood=likelihood, num_outputs=num_outputs)

        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
            
        self.model.train_inputs = [transformed_X]
        if train_Y is not None:
            self.model.train_targets = train_Y.squeeze(-1)

        self.to(train_X)

    def init_inducing_points(
        self,
        inputs: Tensor,
    ) -> Tensor:
        r"""
        Reinitialize the inducing point locations in-place with the current kernel
        applied to `inputs` through the model's inducing point allocation strategy.
        The variational distribution and variational strategy caches are reset.
        Args:
            inputs: (\*batch_shape, n, d)-dim input data tensor.
        Returns:
            (\*batch_shape, m, d)-dim tensor of selected inducing point locations.
        """
        var_strat = self.model.variational_strategy
        clear_cache_hook(var_strat)
        if hasattr(var_strat, "base_variational_strategy"):
            var_strat = var_strat.base_variational_strategy
            clear_cache_hook(var_strat)

        with torch.no_grad():
            num_inducing = var_strat.inducing_points.size(-2)
            inducing_points = self._inducing_point_allocator.allocate_inducing_points(
                inputs=inputs,
                # covar_module=self.model.covar_module,
                covar_module_x=self.model.covar_module_x,
                covar_module_fidelity=self.model.covar_module_fidelity,
                num_inducing=num_inducing,
                input_batch_shape=self._input_batch_shape,
            )
            if inducing_points.shape[0] != var_strat.inducing_points.shape[0]:
                var_strat.inducing_points = inducing_points
            else:
                var_strat.inducing_points.copy_(inducing_points)
            var_strat.variational_params_initialized.fill_(0)

        return inducing_points


class _SingleTaskMultiFidelityVariationalGP(ApproximateGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Optional[Tensor] = None,
        num_outputs: int = 1,
        learn_inducing_points=True,
        # covar_module: Optional[Kernel] = None,
        covar_module_x: Optional[Kernel] = None,
        covar_module_fidelity: Optional[Kernel] = None,
        mean_module: Optional[Mean] = None,
        variational_distribution: Optional[_VariationalDistribution] = None,
        variational_strategy: Type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Optional[Union[Tensor, int]] = None,
        num_fidelity: int = 1,
        inducing_point_allocator=None,
    ) -> None:
        input_batch_shape = train_X.shape[:-2]
        aug_batch_shape = copy.deepcopy(input_batch_shape)
        if num_outputs > 1:
            aug_batch_shape += torch.Size((num_outputs,))
        self._aug_batch_shape = aug_batch_shape

        if mean_module is None:
            mean_module = ConstantMean(batch_shape=self._aug_batch_shape).to(train_X)

        if covar_module_x is None or covar_module_fidelity is None:
            raise NotImplementedError(
                "UserDefinedError: Covariance module of x and fidelity must be provided."
            )

        if inducing_point_allocator is None:
            inducing_point_allocator = GreedyVarianceReduction()

        # initialize inducing points with a pivoted cholesky init if they are not given
        if not isinstance(inducing_points, Tensor):
            raise NotImplementedError(
                "UserDefinedError: Inducing points must be provided."
            )

        if variational_distribution is None:
            variational_distribution = CholeskyVariationalDistribution(
                num_inducing_points=inducing_points.shape[-2],
                batch_shape=self._aug_batch_shape,
            )

        variational_strategy_instance = variational_strategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_points,
        )

        # wrap variational models in independent multi-task variational strategy
        if num_outputs > 1:
            raise NotImplementedError(
                "UserDefinedError: Multi-output in _SingleTaskMultiFidelityVariationalGP not yet supported."
            )
        super().__init__(variational_strategy=variational_strategy_instance)
        self.mean_module = mean_module
        self.covar_module_x = covar_module_x
        self.covar_module_fidelity = covar_module_fidelity

    def forward(self, input) -> MultivariateNormal:
        x = input[..., :-1]
        i = input[..., -1]
        i = i.unsqueeze(-1)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module_x(x)
        covar_fidelity = self.covar_module_fidelity(i)
        covar = covar_x.mul(covar_fidelity)
        latent_dist = MultivariateNormal(mean_x, covar)
        return latent_dist


class SingleTaskMultiFidelityLikeBotorchVariationalGP(ApproximateGPyTorchModel):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Optional[Tensor] = None,
        likelihood: Optional[Likelihood] = None,
        num_outputs: int = 1,
        learn_inducing_points: bool = True,
        covar_module: Optional[Kernel] = None,
        mean_module: Optional[Mean] = None,
        variational_distribution: Optional[_VariationalDistribution] = None,
        variational_strategy: Type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Optional[Union[Tensor, int]] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        inducing_point_allocator: Optional[InducingPointAllocator] = None,
    ) -> None:
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if train_Y is not None:
            if outcome_transform is not None:
                train_Y, _ = outcome_transform(train_Y)
            self._validate_tensor_args(X=transformed_X, Y=train_Y)
            validate_input_scaling(train_X=transformed_X, train_Y=train_Y)
            if train_Y.shape[-1] != num_outputs:
                num_outputs = train_Y.shape[-1]

        self._num_outputs = num_outputs
        self._input_batch_shape = train_X.shape[:-2]
        aug_batch_shape = copy.deepcopy(self._input_batch_shape)
        if num_outputs > 1:
            aug_batch_shape += torch.Size([num_outputs])
        self._aug_batch_shape = aug_batch_shape

        if likelihood is None:
            raise NotImplementedError(
                "Likelihood must be provided while initalising user-defined SingleTaskMultiFidelityVariationalGP."
            )
            if num_outputs == 1:
                noise_prior = GammaPrior(1.1, 0.05)
                noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
                likelihood = GaussianLikelihood(
                    noise_prior=noise_prior,
                    batch_shape=self._aug_batch_shape,
                    noise_constraint=GreaterThan(
                        MIN_INFERRED_NOISE_LEVEL,
                        transform=None,
                        initial_value=noise_prior_mode,
                    ),
                )
            else:
                raise NotImplementedError(
                    "Multitask likelihood should not be used in SingleTaskMultiFidelityVariationalGP."
                )
                likelihood = MultitaskGaussianLikelihood(num_tasks=num_outputs)
        else:
            self._is_custom_likelihood = True

        if inducing_point_allocator is None:
            self._inducing_point_allocator = GreedyVarianceReduction()
        else:
            self._inducing_point_allocator = inducing_point_allocator

        model = _SingleTaskMultiFidelityLikeBotorchVariationalGP(
            train_X=transformed_X,
            train_Y=train_Y,
            num_outputs=num_outputs,
            learn_inducing_points=learn_inducing_points,
            covar_module=covar_module,
            mean_module=mean_module,
            variational_distribution=variational_distribution,
            variational_strategy=variational_strategy,
            inducing_points=inducing_points,
            inducing_point_allocator=self._inducing_point_allocator,
        )

        super().__init__(model=model, likelihood=likelihood, num_outputs=num_outputs)

        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform

        # for model fitting utilities
        # TODO: make this a flag?
        self.model.train_inputs = [transformed_X]
        if train_Y is not None:
            self.model.train_targets = train_Y.squeeze(-1)

        self.to(train_X)

    def init_inducing_points(
        self,
        inputs: Tensor,
    ) -> Tensor:
        r"""
        Reinitialize the inducing point locations in-place with the current kernel
        applied to `inputs` through the model's inducing point allocation strategy.
        The variational distribution and variational strategy caches are reset.
        Args:
            inputs: (\*batch_shape, n, d)-dim input data tensor.
        Returns:
            (\*batch_shape, m, d)-dim tensor of selected inducing point locations.
        """
        var_strat = self.model.variational_strategy
        clear_cache_hook(var_strat)
        if hasattr(var_strat, "base_variational_strategy"):
            var_strat = var_strat.base_variational_strategy
            clear_cache_hook(var_strat)

        with torch.no_grad():
            num_inducing = var_strat.inducing_points.size(-2)
            inducing_points = self._inducing_point_allocator.allocate_inducing_points(
                inputs=inputs,
                covar_module=self.model.covar_module,
                num_inducing=num_inducing,
                input_batch_shape=self._input_batch_shape,
            )
            var_strat.inducing_points.copy_(inducing_points)
            var_strat.variational_params_initialized.fill_(0)

        return inducing_points


class _SingleTaskMultiFidelityLikeBotorchVariationalGP(ApproximateGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Optional[Tensor] = None,
        num_outputs: int = 1,
        learn_inducing_points=True,
        covar_module: Optional[Kernel] = None,
        mean_module: Optional[Mean] = None,
        variational_distribution: Optional[_VariationalDistribution] = None,
        variational_strategy: Type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Optional[Union[Tensor, int]] = None,
        inducing_point_allocator=None,
    ) -> None:
        input_batch_shape = train_X.shape[:-2]
        aug_batch_shape = copy.deepcopy(input_batch_shape)
        if num_outputs > 1:
            aug_batch_shape += torch.Size((num_outputs,))
        self._aug_batch_shape = aug_batch_shape

        if mean_module is None:
            mean_module = ConstantMean(batch_shape=self._aug_batch_shape).to(train_X)

        nu = 2.5
        linear_truncated = True
        data_fidelity = -1
        iteration_fidelity = None

        covar_module, subset_batch_dict = _setup_multifidelity_covar_module(
            dim=train_X.size(-1),
            aug_batch_shape=self._aug_batch_shape,
            data_fidelity=data_fidelity,
            iteration_fidelity=iteration_fidelity,
            linear_truncated=linear_truncated,
            nu=nu,
        )
        self._subset_batch_dict = {
            "likelihood.noise_covar.raw_noise": -2,
            "mean_module.raw_constant": -1,
            "covar_module.raw_outputscale": -1,
            **subset_batch_dict,
        }

        if inducing_point_allocator is None:
            inducing_point_allocator = GreedyVarianceReduction()

        # initialize inducing points with a pivoted cholesky init if they are not given
        if not isinstance(inducing_points, Tensor):
            raise NotImplementedError(
                "UserDefinedError: Inducing points must be provided."
            )

        if variational_distribution is None:
            variational_distribution = CholeskyVariationalDistribution(
                num_inducing_points=inducing_points.shape[-2],
                batch_shape=self._aug_batch_shape,
            )

        variational_strategy_instance = variational_strategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_points,
        )

        # wrap variational models in independent multi-task variational strategy
        if num_outputs > 1:
            raise NotImplementedError(
                "UserDefinedError: Multi-output in _SingleTaskMultiFidelityVariationalGP not yet supported."
            )
        super().__init__(variational_strategy=variational_strategy_instance)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_dist = MultivariateNormal(mean_x, covar_x)
        return latent_dist


def _setup_multifidelity_covar_module(
    dim: int,
    aug_batch_shape: torch.Size,
    iteration_fidelity: Optional[int],
    data_fidelity: Optional[int],
    linear_truncated: bool,
    nu: float,
):
    """Helper function to get the covariance module and associated subset_batch_dict
    for the multifidelity setting.
    Args:
        dim: The dimensionality of the training data.
        aug_batch_shape: The output-augmented batch shape as defined in
            `BatchedMultiOutputGPyTorchModel`.
        iteration_fidelity: The column index for the training iteration fidelity
            parameter (optional).
        data_fidelity: The column index for the downsampling fidelity parameter
            (optional).
        linear_truncated: If True, use a `LinearTruncatedFidelityKernel` instead
            of the default kernel.
        nu: The smoothness parameter for the Matern kernel: either 1/2, 3/2, or
            5/2. Only used when `linear_truncated=True`.
    Returns:
        The covariance module and subset_batch_dict.
    """

    if iteration_fidelity is not None and iteration_fidelity < 0:
        iteration_fidelity = dim + iteration_fidelity
    if data_fidelity is not None and data_fidelity < 0:
        data_fidelity = dim + data_fidelity

    if linear_truncated:
        fidelity_dims = [
            i for i in (iteration_fidelity, data_fidelity) if i is not None
        ]
        kernel = LinearTruncatedFidelityKernel(
            fidelity_dims=fidelity_dims,
            dimension=dim,
            nu=nu,
            batch_shape=aug_batch_shape,
            power_prior=GammaPrior(3.0, 3.0),
        )
    else:
        raise NotImplementedError("LinearTruncated==False not Implemented")
        active_dimsX = [
            i for i in range(dim) if i not in {iteration_fidelity, data_fidelity}
        ]
        kernel = RBFKernel(
            ard_num_dims=len(active_dimsX),
            batch_shape=aug_batch_shape,
            lengthscale_prior=GammaPrior(3.0, 6.0),
            active_dims=active_dimsX,
        )
        additional_kernels = []
        if iteration_fidelity is not None:
            exp_kernel = ExponentialDecayKernel(
                batch_shape=aug_batch_shape,
                lengthscale_prior=GammaPrior(3.0, 6.0),
                offset_prior=GammaPrior(3.0, 6.0),
                power_prior=GammaPrior(3.0, 6.0),
                active_dims=[iteration_fidelity],
            )
            additional_kernels.append(exp_kernel)
        if data_fidelity is not None:
            ds_kernel = DownsamplingKernel(
                batch_shape=aug_batch_shape,
                offset_prior=GammaPrior(3.0, 6.0),
                power_prior=GammaPrior(3.0, 6.0),
                active_dims=[data_fidelity],
            )
            additional_kernels.append(ds_kernel)
        kernel = ProductKernel(kernel, *additional_kernels)

    covar_module = ScaleKernel(
        kernel, batch_shape=aug_batch_shape, outputscale_prior=GammaPrior(2.0, 0.15)
    )

    if linear_truncated:
        subset_batch_dict = {
            "covar_module.base_kernel.raw_power": -2,
            "covar_module.base_kernel.covar_module_unbiased.raw_lengthscale": -3,
            "covar_module.base_kernel.covar_module_biased.raw_lengthscale": -3,
        }
    else:
        subset_batch_dict = {
            "covar_module.base_kernel.kernels.0.raw_lengthscale": -3,
            "covar_module.base_kernel.kernels.1.raw_power": -2,
            "covar_module.base_kernel.kernels.1.raw_offset": -2,
        }
        if iteration_fidelity is not None:
            subset_batch_dict = {
                "covar_module.base_kernel.kernels.1.raw_lengthscale": -3,
                **subset_batch_dict,
            }
            if data_fidelity is not None:
                subset_batch_dict = {
                    "covar_module.base_kernel.kernels.2.raw_power": -2,
                    "covar_module.base_kernel.kernels.2.raw_offset": -2,
                    **subset_batch_dict,
                }

    return covar_module, subset_batch_dict
