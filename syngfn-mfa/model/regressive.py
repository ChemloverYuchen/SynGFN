import torch.nn as nn
from .mlp import ACTIVATION_KEY
import torch
import torch.nn.functional as F
from torch.nn import MSELoss


class RegressiveMLP(nn.Module):
    def __init__(
        self,
        regressive_mode,
        base_num_hidden,
        base_num_layer,
        fid_num_hidden,
        activation,
        dropout_base,
        dropout_fid,
        num_output,
        n_fid,
        config_env,
        device,
        # In OHE, this is num_fid, but in Hartmann this is 1
        feature_dim,
        beta1=0.9,
        beta2=0.999,
        **kwargs,
    ):
        super(RegressiveMLP, self).__init__()
        if config_env.proxy_state_format == "ohe":
            self.num_fid_parameter = n_fid
            input_classes = config_env.length  # length
            input_max_length = config_env.n_dim  # n_dim
            self.init_layer_depth = int((input_classes) * (input_max_length))
        else:
            self.init_layer_depth = config_env.n_dim
            self.num_fid_parameter = 1
        self.feature_dim = feature_dim
        self.n_fid = n_fid

        self.fid_num_hidden = fid_num_hidden
        self.num_output = num_output
        self.activation = ACTIVATION_KEY[activation]
        self.device = device
        self.base_hidden_layers = [base_num_hidden] * base_num_layer
        self.is_fid_param = True
        base_layers = [
            nn.Linear(self.init_layer_depth, self.base_hidden_layers[0]),
            self.activation,
        ]
        base_layers += [nn.Dropout(dropout_base)]
        for i in range(1, len(self.base_hidden_layers)):
            base_layers.extend(
                [
                    nn.Linear(
                        self.base_hidden_layers[i - 1], self.base_hidden_layers[i]
                    ),
                    self.activation,
                    nn.Dropout(dropout_base),
                ]
            )
        if base_layers != []:
            base_layers.append(nn.Linear(self.base_hidden_layers[-1], self.num_output))
            self.base_module = nn.Sequential(*base_layers)

        # list of layers created
        if regressive_mode == "seq":
            self.get_features = self.get_feature_seq_regressive
            self.layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(
                            self.init_layer_depth + self.num_output,
                            self.fid_num_hidden,
                        ),
                        nn.Dropout(dropout_fid),
                        self.activation,
                        nn.Linear(self.fid_num_hidden, self.fid_num_hidden),
                        nn.Dropout(dropout_fid),
                        self.activation,
                        nn.Linear(self.fid_num_hidden, self.fid_num_hidden),
                        nn.Dropout(dropout_fid),
                        # self.activation,
                        # nn.Linear(self.fid_num_hidden, self.num_output),
                    )
                    for i in range(self.n_fid)
                ]
            )
            self.forward = self.forward_seq_regressive
        elif regressive_mode == "full":
            self.get_features = self.get_feature_full_regressive
            self.layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(
                            self.init_layer_depth + (self.num_output + i),
                            self.fid_num_hidden,
                        ),
                        nn.Dropout(dropout_fid),
                        self.activation,
                        nn.Linear(self.fid_num_hidden, self.fid_num_hidden),
                        nn.Dropout(dropout_fid),
                        self.activation,
                        nn.Linear(self.fid_num_hidden, self.fid_num_hidden),
                        nn.Dropout(dropout_fid),
                    )
                    for i in range(self.n_fid)
                ]
            )
            self.forward = self.forward_full_regressive
            # self.get_feature = self.get_feature_full_regressive
        self.output_layer = nn.Sequential(
            self.activation,
            nn.Linear(self.fid_num_hidden, self.num_output),
        )
        self.betas = (beta1, beta2)

    def get_feature_seq_regressive(self, input):
        """
        Have not performed experiments with this.
        """
        x, fid = (
            input[:, : -self.num_fid_parameter],
            input[:, -self.num_fid_parameter :],
        )
        if self.num_fid_parameter != self.n_fid:
            fid = fid.squeeze(-1)
            fid_ohe = F.one_hot(fid.long(), num_classes=self.n_fid + 1)[:, :-1].to(
                torch.float32
            )
        output_interm = self.base_module(x)
        features = []
        for m in range(self.n_fid):
            # TODO: MIGHT BREAK BECAUSE OUTPUT_INTERM IS EMPTY
            augment_input = torch.cat([output_interm, x], axis=1)
            feature = self.layers[m](augment_input)
            output_interm = self.output_layer(feature)
            features.append(feature)
        feature_tensor = torch.stack(features, dim=1).to(self.device)
        feature_masked = feature_tensor[fid == 1]
        return feature_masked

    def get_feature_full_regressive(self, input):
        x, fid = (
            input[:, : -self.num_fid_parameter],
            input[:, -self.num_fid_parameter :],
        )
        if self.num_fid_parameter != self.n_fid:
            fid = fid.squeeze(-1)
            fid_ohe = F.one_hot(fid.long(), num_classes=self.n_fid + 1)[:, :-1].to(
                torch.float32
            )
        # input, fid_ohe = self.preprocess(input, fid)
        if self.base_module is not None:
            output_interm = self.base_module(x)
            augment_input = torch.cat([output_interm, x], axis=1)
        else:
            augment_input = x
        features = []
        for m in range(self.n_fid):
            feature = self.layers[m](augment_input)
            output_fid = self.output_layer(feature)
            augment_input = torch.cat([output_fid, augment_input], axis=1)
            features.append(feature)
        feature_tensor = torch.stack(features, dim=1).to(self.device)
        feature_masked = feature_tensor[fid_ohe == 1]
        return feature_masked

    def forward_seq_regressive(self, input):
        """Implementation of DNN-MFBO
        Args:
            input: one hot encoded input tensor
            fid: tensor of integers representing the fidelity
                example: tensor([2, 1])
        Returns:
            output_masked: tensor of shape (batch_size, num_output=1)"""
        feature = self.get_feature_seq_regressive(input)
        output = self.output_layer(feature)
        return output

    def forward_full_regressive(self, input):
        """Implementation of BMBO-DARN
        Args:
            input: one hot encoded input tensor
            fid: tensor of integers representing the fidelity
                example: tensor([2, 1])
        Returns:
            output_masked: tensor of shape (batch_size, num_output=1)"""
        feature = self.get_feature_full_regressive(input)
        output = self.output_layer(feature)
        return output

    def param_groups(self, lr, weight_decay):
        shared_group = dict(
            params=[], lr=lr, weight_decay=weight_decay, betas=self.betas
        )
        for p_name, param in self.named_parameters():
            shared_group["params"].append(param)
        return [shared_group]

    def train_step(
        self, input_batch, optimizer, target_batch, criterion=MSELoss(), **kwargs
    ):
        """
        Required so that one DKL can work with many models
        """
        self.train()
        optimizer.zero_grad()
        output = self(input_batch)
        loss = criterion(output, target_batch.unsqueeze(-1))
        loss.backward()
        optimizer.step()
        return loss

    def eval_epoch(self, loader, criterion=MSELoss(), **kwargs):
        metrics = {"loss": 0.0}
        self.eval()
        with torch.no_grad():
            for (input_batch, target_batch) in loader:
                input_batch = input_batch.to("cuda")
                target_batch = target_batch.to("cuda")
                loss = criterion(self(input_batch), target_batch.unsqueeze(-1))
                metrics["loss"] += loss.item() / len(loader)
        metrics = {f"test_{key}": val for key, val in metrics.items()}
        return metrics
