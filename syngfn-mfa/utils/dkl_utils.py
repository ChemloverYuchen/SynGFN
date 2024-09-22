import torch
import numpy as np


def batched_call(
    fn, arg_array, batch_size, *args, **kwargs
):
    batch_size = (
        arg_array.shape[0] if batch_size is None else batch_size
    )
    num_batches = max(1, arg_array.shape[0] // batch_size)

    if isinstance(
        arg_array, np.ndarray
    ):
        arg_batches = np.array_split(arg_array, num_batches)
    elif isinstance(arg_array, torch.Tensor):
        arg_batches = torch.split(arg_array, num_batches)
    else:
        raise ValueError
    return [fn(batch, *args, **kwargs) for batch in arg_batches]


def float_to_long(x: torch.Tensor):
    return torch.LongTensor(x).to(x.device)
