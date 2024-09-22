import math
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import hydra
from model.mlm import sample_mask
from .masked_layers import mResidualBlock
from gflownet.gflownet.utils.common import set_device, set_float_precision


class LanguageModel(nn.Module):
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    def __init__(
        self,
        model,
        batch_size,
        num_epochs,
        patience,
        lr,
        max_shift,
        mask_ratio,
        tokenizer,
        device,
        float_precision,
        feature_dim,
        **kwargs
    ):
        super().__init__()
        self.device = device
        self.float = float_precision
        self.feature_dim = feature_dim
        config_model = model
        self.model = hydra.utils.instantiate(
            config_model, tokenizer=tokenizer, **kwargs
        )  # Change max_len of model
        self.model = self.model.to(self.device)
        # self.model.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.lr = lr
        self.max_shift = max_shift
        self.tokenizer = tokenizer
        self.mask_ratio = mask_ratio
        self.is_fid_param = False

    def forward(self, inputs):
        return self.model(inputs)

    def get_features(self, inputs):
        return self(inputs)
        # fid_array = inputs[..., -1].to(self.device)
        return self.model(inputs)

    def train_step(self, optimizer, input_batch, n_fid, loss_scale=1.0, **kwargs):
        optimizer.zero_grad(set_to_none=True)
        mask_ratio = self.mask_ratio
        if n_fid > 1:
            input_batch = input_batch[..., :-1]
        # replace random tokens with mask token
        mask_idxs = sample_mask(input_batch, self.tokenizer, mask_ratio)
        masked_token_batch = input_batch.clone().to(self.device)
        np.put_along_axis(
            masked_token_batch, mask_idxs, self.tokenizer.masking_idx, axis=1
        )

        # get predicted logits for masked tokens
        logits, _ = self.logits_from_tokens(masked_token_batch)
        vocab_size = logits.shape[-1]
        masked_logits = np.take_along_axis(logits, mask_idxs[..., None], axis=1).view(
            -1, vocab_size
        )

        # use the ground-truth tokens as labels
        masked_tokens = np.take_along_axis(input_batch, mask_idxs, axis=1)
        masked_tokens = masked_tokens.view(-1).to(self.device)  # torch.Size([160])

        loss = loss_scale * F.cross_entropy(masked_logits, masked_tokens)
        loss.backward()
        optimizer.step()

        return loss  # masked_logits, masked_tokens

    def pool_features(self, src_tok_features, src_mask):
        lat_tok_features, pooled_features = self.model.function_head(
            src_tok_features, padding_mask=src_mask, pooling_mask=src_mask
        )
        return lat_tok_features, pooled_features

    def get_token_features(self, src_tok_idxs):
        src_tok_features, src_mask = self.model.enc_tok_features(src_tok_idxs)
        return src_tok_features, src_mask

    def logits_from_tokens(self, src_tok_idxs):
        src_tok_features, src_mask = self.get_token_features(src_tok_idxs)
        tgt_tok_logits, tgt_mask = self.logits_from_features(
            src_tok_features, src_mask, lat_tok_features=None
        )
        return tgt_tok_logits, tgt_mask

    def logits_from_features(
        self, src_tok_features, src_mask, lat_tok_features, tgt_lens=None
    ):
        lat_tok_features, tgt_tok_features, tgt_mask, _ = self.model.dec_tok_features(
            src_tok_features, src_mask, lat_tok_features, tgt_lens
        )
        tgt_tok_logits = self.model.tgt_tok_logits(tgt_tok_features)
        return tgt_tok_logits, tgt_mask

    def param_groups(self, *args, **kwargs):
        return self.model.param_groups(*args, **kwargs)


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        batch_first=False,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len + 1).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len + 1, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        if self.batch_first:
            pe = pe.transpose(1, 0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] or [batch_size, seq_len, embedding_dim]
        """
        if self.batch_first:
            x = x + self.pe[:, : x.size(1)]
        else:
            x = x + self.pe[: x.size(0)]

        return self.dropout(x)


class FunctionHead(nn.Module):
    def __init__(
        self, input_dim, output_dim, kernel_size, layernorm, dropout_p, num_heads, type
    ):
        super().__init__()
        if type == "conv":
            self.att_layer = mResidualBlock(
                input_dim, input_dim, kernel_size, layernorm, dropout_p
            )
        elif type == "mha":
            self.att_layer = nn.MultiheadAttention(
                input_dim, num_heads, dropout=dropout_p, batch_first=True
            )
            self.dropout = nn.Dropout(dropout_p)
        else:
            raise ValueError
        self.type = type
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(
        self,
        tok_features: Tensor,
        padding_mask: Tensor,
        pooling_mask: Tensor,
    ):

        if self.type == "conv":
            tok_features, _ = self.att_layer(
                (tok_features.permute(0, 2, 1), padding_mask)
            )
            tok_features = tok_features.permute(0, 2, 1)  # torch.Size([426, 36, 16])
        else:
            key_padding_mask = ~padding_mask.bool()  # invert mask
            tok_features, _ = self.att_layer(
                tok_features,
                tok_features,
                tok_features,
                key_padding_mask,
                need_weights=False,
            )
            tok_features = self.dropout(F.gelu(tok_features))

        pooling_mask = pooling_mask.unsqueeze(-1).float()
        pooled_features = (pooling_mask * tok_features).sum(-2) / (
            pooling_mask.sum(-2) + 1e-6
        )
        pooled_features = self.linear(pooled_features)
        return tok_features, pooled_features


class LengthHead(nn.Module):
    def __init__(self, input_dim, max_len_delta):
        super().__init__()
        num_classes = max_len_delta * 2 + 1
        self.linear = nn.Linear(input_dim, num_classes)
        self.max_len_delta = max_len_delta

    def forward(self, tok_features, pooling_mask):
        pooling_mask = pooling_mask.unsqueeze(-1).float()
        pooled_features = (pooling_mask * tok_features).sum(-2) / (
            pooling_mask.sum(-2) + 1e-6
        )
        logits = self.linear(pooled_features)
        return logits

    def sample(self, src_lens, logits):
        if self.max_len_delta == 0:
            return src_lens
        tgt_len_dist = torch.distributions.Categorical(logits=logits)
        len_deltas = tgt_len_dist.sample()
        len_deltas -= self.max_len_delta
        return src_lens + len_deltas


class LengthTransform(nn.Module):
    """
    monotonic location-based attention mechanism from
    https://arxiv.org/abs/1908.07181
    """

    def __init__(self):
        super().__init__()
        self.register_parameter("lengthscale", nn.Parameter(torch.tensor(1.0)))

    def forward(
        self,
        src_tok_features: Tensor,
        src_mask: Tensor,
        tgt_lens: Tensor,
    ):

        batch_size, num_src_tokens = src_mask.shape
        src_lens = src_mask.float().sum(-1)
        tgt_lens = tgt_lens.to(src_lens)

        if torch.all(
            src_lens == tgt_lens
        ):  # Always enters here in eval and mlm_train_step
            return src_tok_features, src_mask.bool()
        else:
            raise NotImplementedError(
                "Length Transformation nto supported here. Check Lambo codebase."
            )
