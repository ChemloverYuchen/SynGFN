from botorch.models.utils.inducing_point_allocators import (
    InducingPointAllocator,
    NEG_INF,
    _pivoted_cholesky_init,
    UnitQualityFunction,
)
import torch


class MultiFidelityInducingPointAllocator(InducingPointAllocator):
    def allocate_inducing_points(
        self,
        inputs,
        covar_module_x,
        covar_module_fidelity,
        num_inducing,
        input_batch_shape,
    ):
        r"""
        Initialize the `num_inducing` inducing point locations according to a
        specific initialization strategy. todo say something about quality
        Args:
            inputs: A (\*batch_shape, n, d)-dim input data tensor.
            covar_module: GPyTorch Module returning a LinearOperator kernel matrix.
            num_inducing: The maximun number (m) of inducing points (m <= n).
            input_batch_shape: The non-task-related batch shape.
        Returns:
            A (\*batch_shape, m, d)-dim tensor of inducing point locations.
        """
        quality_function = self._get_quality_function()
        covar_module_x = covar_module_x.to(inputs.device)
        covar_module_fidelity = covar_module_fidelity.to(inputs.device)

        x_input = inputs[..., :-1]
        f_input = inputs[..., -1].unsqueeze(-1)
        x_output = covar_module_x(x_input)
        f_output = covar_module_fidelity(f_input)
        output = x_output.mul(f_output)
        train_train_kernel = output.evaluate_kernel()

        # base case
        if train_train_kernel.ndimension() == 2:
            quality_scores = quality_function(inputs)
            inducing_points = _pivoted_cholesky_init(
                train_inputs=inputs,
                kernel_matrix=train_train_kernel,
                max_length=num_inducing,
                quality_scores=quality_scores,
                epsilon=1e-20,
            )
        else:
            raise NotImplementedError("Not copied from Botorch yet")

        return inducing_points


class MultiFidelityGreedyVarianceReduction(MultiFidelityInducingPointAllocator):
    r"""
    The inducing point allocator proposed by [burt2020svgp]_, that
    greedily chooses inducing point locations with maximal (conditional)
    predictive variance.
    """

    def _get_quality_function(
        self,
    ):
        """
        Build the unit quality function required for the greedy variance
        reduction inducing point allocation strategy.
        Returns:
            A quality function.
        """

        return UnitQualityFunction()
