import torch.nn as nn
import torch
import math

ACTIVATION_KEY = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "leaky_relu": nn.LeakyReLU(),
}


class MLP(nn.Module):
    def __init__(
        self,
        n_hid,
        n_layers,
        num_output,
        dropout_prob,
        activation,
        feature_dim,
        beta1=0.9,
        beta2=0.999,
        transformerCall=False,
        config_env=None,
        input_dim=None,
        **kwargs,
    ):
        super(MLP, self).__init__()
        self.feature_dim = feature_dim
        """
        Args:
            model parameters
            transformerCall: boolean to check if MLP is initialised within transformer
            if transformerCall=True, 
                input_dim is passed as a parameter
            else
                configuration of env (required for specifying input dimensions of model) is passed

        """
        self.activation = ACTIVATION_KEY[activation]
        self.is_fid_param = False

        # TODO: this is grid specific for now, make it general (for apatamers and torus)
        if config_env.proxy_state_format == "ohe":
            self.input_max_length = config_env.n_dim
            self.input_classes = config_env.length
        else:
            self.input_classes = 1
            self.input_max_length = config_env.n_dim
        self.out_dim = num_output

        if transformerCall == False:
            self.hidden_layers = [n_hid] * n_layers
            self.dropout_prob = dropout_prob
            self.init_layer_depth = int((self.input_classes) * (self.input_max_length))

        else:
            # this clause is entered only when transformer specific config is passed
            self.hidden_layers = [self.mlp.hidden_dim] * self.mlp.num_layer
            self.dropout_prob = self.mlp.dropout_prob
            self.init_layer_depth = self.embed_dim

        layers = [
            nn.Linear(self.init_layer_depth, self.hidden_layers[0]),
            self.activation,
        ]
        layers += [nn.Dropout(self.dropout_prob)]
        for i in range(1, len(self.hidden_layers)):
            layers.extend(
                [
                    nn.Linear(self.hidden_layers[i - 1], self.hidden_layers[i]),
                    self.activation,
                    nn.Dropout(self.dropout_prob),
                ]
            )
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(self.hidden_layers[-1], self.out_dim)
        self.betas = (beta1, beta2)

        # layers.append(nn.Linear(self.hidden_layers[-1], self.out_dim))

    def forward(self, x, **kwargs):
        """
        Args:
            tensor of dimension batch x max_seq_length_in_batch x k,
            where k is num_tokens if input is required to be one-hot encoded

        """
        # Pads the tensor till maximum length of dataset
        # input = torch.zeros(x.shape[0], self.init_layer_depth).to(x.device)
        # input[:, : x.shape[1]] = x
        # Performs a forward call
        feature = self.feature_extractor(x)
        output = self.output_layer(feature)
        return output

    def get_features(self, x, **kwargs):
        feature = self.feature_extractor(x)
        return feature

    def param_groups(self, lr, weight_decay):
        shared_group = dict(
            params=[], lr=lr, weight_decay=weight_decay, betas=self.betas
        )
        for p_name, param in self.named_parameters():
            shared_group["params"].append(param)
        return [shared_group]

    def train_step(
        self,
        input_batch,
        optimizer,
        target_batch,
        n_fid=1,
        criterion=nn.MSELoss(),
        **kwargs,
    ):
        optimizer.zero_grad()
        if n_fid > 1:
            input_batch = input_batch[..., :-1]
        output = self(input_batch)
        loss = criterion(output, target_batch.unsqueeze(-1))
        # self.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def eval_epoch(self, loader, criterion=nn.MSELoss(), n_fid=1, **kwargs):

        metrics = {"loss": 0.0}
        self.eval()
        with torch.no_grad():
            for (x, y) in loader:
                x = x.to("cuda")
                if n_fid > 1:
                    x = x[..., :-1]
                y = y.to("cuda").unsqueeze(-1)
                y_hat = self(x)
                loss = criterion(y_hat, y)
                metrics["loss"] += loss.item() / len(loader)
        metrics = {f"test_{key}": val for key, val in metrics.items()}
        return metrics


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_layer,
        num_output,
        dropout_prob,
        num_head,
        pre_ln,
        factor,
        activation,
        mlp,
        config_env,
    ):
        super(Transformer, self).__init__()
        self.input_classes = config_env.n_dim  # number of classes
        self.input_max_length = config_env.length  # sequence length
        self.num_hidden = hidden_dim
        self.dropout_prob = dropout_prob
        self.num_head = num_head
        self.pre_ln = pre_ln
        self.num_layer = num_layer
        self.factor = factor
        self.num_output = num_output
        self.pos = PositionalEncoding(
            self.num_hidden,
            dropout=self.dropout_prob,
            max_len=self.input_max_length + 2,
        )
        self.embedding = nn.Embedding(self.input_classes + 1, self.num_hidden)

        encoder_layers = nn.TransformerEncoderLayer(
            self.num_hidden,
            self.num_head,
            self.num_hidden,
            dropout=self.dropout_prob,
            norm_first=self.pre_ln,
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, self.num_layer)
        self.output = MLP(
            self.num_hidden * self.factor,
            mlp.num_layer,
            self.num_output,
            mlp.dropout_prob,
            activation,
            transformerCall=True,
            input_dim=self.num_hidden,
        )
        self.logsoftmax2 = torch.nn.LogSoftmax(2)

    def model_params(self):
        return (
            list(self.pos.parameters())
            + list(self.embedding.parameters())
            + list(self.encoder.parameters())
            + list(self.output.parameters())
        )

    def forward(self, x, mask=None):
        """
        Args:
            - x: float tensor of dimension batch x max_seq_length_in_batch
        Example
            - x is of dim 32x2 for a 3x3 grid
        """
        x = x.to(torch.int64)
        x = torch.transpose(x, 1, 0)
        x = self.embedding(x)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        pooled_x = x[0, torch.arange(x.shape[1])]
        y = self.output(pooled_x)
        return y
