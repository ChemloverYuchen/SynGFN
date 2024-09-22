import math
import numpy as np
import torch
from torch.nn import functional as F
from torch import LongTensor


def sample_mask(
    token_batch: LongTensor, tokenizer, mask_ratio: float = 0.125, mask_size=None
):
    """Chooses the lements ot be masked but ensures that none of the special tokens are masked.
    Args:
            token_batch: (batch_size, num_tokens)
            tokenizer: only necessary to avoid masking special tokens
            mask_ratio: proportion of tokens to mask
            mask_size: (optional) override mask_ratio with a specific mask size
    Returns:
            mask_idxs: (batch_size, mask_size) np.ndarray of position indexes to mask
    """
    if mask_size is None:  # mask_Size = 5
        mask_size = math.ceil(token_batch.shape[-1] * mask_ratio)

    special_idxs = torch.tensor(tokenizer.special_idxs).view(-1, 1, 1).to(token_batch)
    is_non_special = token_batch.ne(special_idxs).prod(dim=0).float()
    mask_weights = is_non_special / is_non_special.sum(dim=-1, keepdims=True)
    mask_idxs = torch.multinomial(mask_weights, mask_size, replacement=False)
    return mask_idxs.detach().cpu().numpy()


def mlm_eval_epoch(model, loader, mask_ratio, n_fid=1):
    metrics = dict(
        perplexity=0.0,
    )
    model.eval()
    # print("\nUser-Defined Warning: Converting states in test loader to integer for mlm evaluation.")
    for minibatch in loader:
        if isinstance(minibatch, tuple) or isinstance(minibatch, list):
            token_batch = minibatch[0]
        # elif :

        else:
            assert torch.is_tensor(minibatch)
            token_batch = minibatch
        if n_fid > 1:
            token_batch = token_batch[..., :-1]
        token_batch = token_batch.long()
        # token_batch is padded states
        # replace random tokens with mask token
        mask_idxs = sample_mask(token_batch, model.tokenizer, mask_ratio)
        masked_token_batch = token_batch.clone().to(model.device)
        np.put_along_axis(
            masked_token_batch, mask_idxs, model.tokenizer.masking_idx, axis=1
        )

        # get predicted logits for masked tokens
        logits, _ = model.logits_from_tokens(masked_token_batch)
        vocab_size = logits.shape[-1]
        masked_logits = np.take_along_axis(logits, mask_idxs[..., None], axis=1).view(
            -1, vocab_size
        )

        # use the ground-truth tokens as labels
        masked_tokens = np.take_along_axis(token_batch, mask_idxs, axis=1)
        masked_tokens = masked_tokens.view(-1).to(model.device)

        # logging
        log_prob = F.log_softmax(masked_logits, dim=-1)
        log_prob = np.take_along_axis(
            log_prob, masked_tokens.cpu().numpy()[..., None], axis=1
        )
        metrics["perplexity"] += 2 ** (-(log_prob / math.log(2)).mean().detach()) / len(
            loader
        )

    metrics = {key: val.item() for key, val in metrics.items()}
    metrics = {f"test_{key}": val for key, val in metrics.items()}

    return metrics
