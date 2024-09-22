from __future__ import annotations
import torch
from omegaconf import OmegaConf
from torch import nn

from gflownet.utils.common import set_device, set_float_precision

from gflownet.envs.building_block import BuildingBlock, smiles_list

class Policy:
    def __init__(
        self, 
        config, 
        env: BuildingBlock, 
        device, 
        float_precision, 
        base: 'Policy'=None
        ):
        # If config is null, default to uniform
        if config is None:
            config = OmegaConf.create()
            config.type = "uniform"
        # Device and float precision
        self.device = set_device(device)
        self.float = set_float_precision(float_precision)
        # Input and output dimensions
        self.state_dim_1 = env.policy_input_dim_1
        self.state_dim_2 = env.policy_input_dim_2
        
        self.checkpoint = config.checkpoint
        
        
        self.fixed_output_1 = torch.tensor(env.fixed_policy_output_1).to(
            dtype=self.float, device=self.device
        )
        self.fixed_output_2 = torch.tensor(env.fixed_policy_output_2).to(
            dtype=self.float, device=self.device
        )
        self.output_dim_1 = len(self.fixed_output_1)
        self.output_dim_2 = len(self.fixed_output_2)
        
        self.random_output_1 = torch.tensor(env.random_policy_output_1).to(
            dtype=self.float, device=self.device
        )
        
        self.random_output_2 = torch.tensor(env.random_policy_output_2).to(
            dtype=self.float, device=self.device
        )
        
        self.vocab = smiles_list+["EOS"]+["UNI"]
        
        
        if "shared_weights" in config:#for backward_policy
            self.shared_weights = config.shared_weights
        else:
            self.shared_weights = False
            
        self.base = base
        
        if "n_hid" in config:
            self.n_hid = config.n_hid
        else:
            self.n_hid = None
            
        if "n_layers" in config:
            self.n_layers = config.n_layers
        else:
            self.n_layers = None
            
        if "tail" in config:
            self.tail = config.tail
        else:
            self.tail = []
            
        if "type" in config:
            self.type = config.type
        elif self.shared_weights:
            self.type = self.base.type
        else:
            raise "Policy type must be defined if shared_weights is False"
        
        if "latent_dim" in config:
            self.latent_dim = config.latent_dim
        else:
            self.latent_dim = 512
        if "padding_idx_1" in config:
            self.padding_idx_1 = config.padding_idx_1
        else:
            self.padding_idx_1 = 0
            
        # Instantiate policy
        if self.type == "fixed_1":
            self.model = self.fixed_distribution_1
            self.is_model = False
        elif self.type == "fixed_2":
            self.model = self.fixed_distribution_2
            self.is_model = False
        elif self.type == "random_1":
            self.model = self.random_distribution_1
            self.is_model = False
        elif self.type == "random_2":
            self.model = self.random_distribution_2
            self.is_model = False
        elif self.type == "uniform_1":
            self.model = self.uniform_distribution_1
            self.is_model = False
        elif self.type == "uniform_2":
            self.model = self.uniform_distribution_2
            self.is_model = False
        elif self.type == "policy_1":
            self.model = self.make_mlp_1(nn.LeakyReLU())
            self.is_model = True
        elif self.type == "policy_2":
            self.model = self.make_mlp_2(nn.LeakyReLU())
            self.is_model = True
        else:
            raise "Policy model type not defined"
        if self.is_model:
            self.model.to(self.device)
            
    def __call__(self, states):
        return self.model(states)

    def make_mlp_1(self, activation):
        """
        Defines an MLP with no top layer activation
        If share_weight == True,
            baseModel (the model with which weights are to be shared) must be provided
        Args
        ----
        layers_dim : list
            Dimensionality of each layer
        activation : Activation
            Activation function
        """
        if self.shared_weights == True and self.base is not None:
            mlp = nn.Sequential(
                self.base.model[:-1],
                nn.Linear(
                    self.base.model[-1].in_features, self.base.model[-1].out_features
                ),
            )
            return mlp.to(dtype=self.float)

        elif self.shared_weights == False:
            layers_dim = (
                [self.state_dim_1] + [self.n_hid] * self.n_layers + [(self.output_dim_1)]
            )
            mlp = nn.Sequential(
                *(
                    sum(
                        [
                            [nn.Linear(idim, odim)]
                            + ([activation] if n < len(layers_dim) - 2 else [])
                            + ([nn.Dropout(p=0.1)] if n == len(layers_dim) - 3 else [])
                            for n, (idim, odim) in enumerate(zip(layers_dim, layers_dim[1:]))
                        ],
                        [],
                    )
                    + self.tail
                )
            )

            return mlp.to(dtype=self.float)
        else:
            raise ValueError(
                "Base Model must be provided when shared_weights is set to True"
            )
            
    def make_mlp_2(self, activation):
        """
        Defines an MLP with no top layer activation
        If share_weight == True,
            baseModel (the model with which weights are to be shared) must be provided
        Args
        ----
        layers_dim : list
            Dimensionality of each layer
        activation : Activation
            Activation function
        """
        if self.shared_weights == True and self.base is not None:
            mlp = nn.Sequential(
                self.base.model[:-1],
                nn.Linear(
                    self.base.model[-1].in_features, self.base.model[-1].out_features
                ),
            )
            return mlp.to(dtype=self.float)

        elif self.shared_weights == False:
            layers_dim = (
                [self.state_dim_2] + [self.n_hid] * self.n_layers + [(self.output_dim_2)]
            )
            mlp = nn.Sequential(
                *(
                    sum(
                        [
                            [nn.Linear(idim, odim)]
                            + ([activation] if n < len(layers_dim) - 2 else [])
                            for n, (idim, odim) in enumerate(
                                zip(layers_dim, layers_dim[1:])
                            )
                        ],
                        [],
                    )
                    + self.tail
                )
            )
            return mlp.to(dtype=self.float)
        else:
            raise ValueError(
                "Base Model must be provided when shared_weights is set to True"
            )
            
    def fixed_distribution_1(self, states):
        """
        Returns the fixed distribution specified by the environment.
        Args: states: tensor
        """
        return torch.tile(self.fixed_output_1, (len(states), 1)).to(
            dtype=self.float, device=self.device
        )
    def fixed_distribution_2(self, states):
        """
        Returns the fixed distribution specified by the environment.
        Args: states: tensor
        """
        return torch.tile(self.fixed_output_2, (len(states), 1)).to(
            dtype=self.float, device=self.device
        )
    def random_distribution_1(self, states):
        """
        Returns the random distribution specified by the environment.
        Args: states: tensor
        """
        return torch.tile(self.random_output_1, (len(states), 1)).to(
            dtype=self.float, device=self.device
        )
    def random_distribution_2(self, states):
        """
        Returns the random distribution specified by the environment.
        Args: states: tensor
        """
        return torch.tile(self.random_output_2, (len(states), 1)).to(
            dtype=self.float, device=self.device
        )
    def uniform_distribution_1(self, states):
        """
        Return action logits (log probabilities) from a uniform distribution
        Args: states: tensor
        """
        return torch.ones(
            (len(states), self.output_dim_1), dtype=self.float, device=self.device
        )
    def uniform_distribution_2(self, states):
        """
        Return action logits (log probabilities) from a uniform distribution
        Args: states: tensor
        """
        return torch.ones(
            (len(states), self.output_dim_2), dtype=self.float, device=self.device
        )