import torch
import torch.nn as nn


class Apply(nn.Module):
    def __init__(self, module, dim=0):
        super().__init__()
        self.module = module
        self.dim = dim

    def forward(self, x):
        xs = list(x)
        xs[self.dim] = self.module(xs[self.dim])
        return xs


# class LayerNorm1d(nn.BatchNorm1d):
#     """n-dimensional batchnorm that excludes points outside the mask from the statistics"""

#     def forward(self, x):
#         """input (B,c,n), computes statistics averaging over c and n"""
#         sum_dims = list(range(len(x.shape)))[1:]

#         xmean = x.mean(dim=sum_dims, keepdims=True)
#         xxmean = (x * x).mean(dim=sum_dims, keepdims=True)
#         var = xxmean - xmean * xmean
#         std = var.clamp(self.eps) ** 0.5
#         ratio = self.weight[:, None] / std
#         output = x * ratio + (self.bias[:, None] - xmean * ratio)
#         return output


class MaskLayerNorm1d(nn.LayerNorm):
    """
    Custom masked layer-norm layer
    """

    def forward(self, inp: tuple):
        x, mask = inp

        batch_size, num_channels, num_tokens = x.shape
        reshaped_mask = mask[:, None]

        sum_dims = list(range(len(x.shape)))[1:]
        xsum = x.sum(dim=sum_dims, keepdims=True)
        xxsum = (x * x).sum(dim=sum_dims, keepdims=True)
        numel_notnan = reshaped_mask.sum(dim=sum_dims, keepdims=True) * num_channels

        xmean = xsum / numel_notnan
        xxmean = xxsum / numel_notnan

        var = xxmean - (xmean * xmean)
        std = var.clamp(self.eps) ** 0.5
        ratio = self.weight / std

        output = (x - xmean) * ratio + self.bias

        return output, mask


@torch.jit.script
def fused_swish(x):
    return x * x.sigmoid()


class mResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        layernorm,
        dropout_p=0.1,
        act_fn="swish",
        stride=1,
    ):
        super().__init__()
        self.conv_1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding="same",
            stride=stride,
            bias=False,
        )
        self.conv_2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding="same", stride=stride
        )
        if layernorm:
            self.norm_1 = MaskLayerNorm1d(normalized_shape=[in_channels, 1])
            self.norm_2 = MaskLayerNorm1d(normalized_shape=[out_channels, 1])
        # else:
        #     self.norm_1 = MaskBatchNormNd(in_channels)
        #     self.norm_2 = MaskBatchNormNd(out_channels)

        if act_fn == "swish":
            self.act_fn = fused_swish
        else:
            self.act_fn = nn.ReLU(inplace=True)

        if not in_channels == out_channels:
            self.proj = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, padding="same", stride=1
            )
        else:
            self.proj = None

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, inputs):
        # assumes inputs are already properly masked
        resid, mask = inputs
        x, _ = self.norm_1((resid, mask))
        x = mask[:, None] * x
        x = self.act_fn(x)
        x = self.conv_1(x)
        x = mask[:, None] * x

        x, _ = self.norm_2((x, mask))
        x = mask[:, None] * x
        x = self.act_fn(x)
        x = self.conv_2(x)

        if self.proj is not None:
            resid = self.proj(resid)

        x = mask[:, None] * (x + resid)

        return self.dropout(x), mask
