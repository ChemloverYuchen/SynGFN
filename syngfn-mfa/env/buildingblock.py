from gflownet.gflownet.envs.building_block import BuildingBlock as GflowNetBuildingBlock
import torch
from torchtyping import TensorType
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import random
import ast
import itertools

class BuildingBlock(GflowNetBuildingBlock):
    def __init__(
        self, proxy_state_format, beta_factor=0, norm_factor=1, proxy=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.proxy_state_format = proxy_state_format
        self.beta_factor = beta_factor
        self.norm_factor = norm_factor
        if proxy is not None:
            self.set_proxy(proxy)
        if self.proxy_state_format == "oracle":
            self.statebatch2proxy = self.statebatch2oracle
        elif self.proxy_state_format == "state":
            self.statebatch2proxy = self.statebatch2state
            self.statetorch2proxy = self.statetorch2state
        else:
            raise ValueError(
                "Invalid proxy_state_format: {}".format(self.proxy_state_format)
            )
        self.tokenizer = None
    
    def set_proxy(self, proxy):
        self.proxy = proxy
        if hasattr(self, "proxy_factor"):
            return
        if self.proxy is not None and self.proxy.maximize is not None:
            maximize = self.proxy.maximize
        elif self.oracle is not None:
            maximize = self.oracle.maximize
        else:
            raise ValueError("Proxy and Oracle cannot be None together.")
        if maximize:
            self.proxy_factor = 1.0
        else:
            self.proxy_factor = -1.0
            
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def unpad_function(self, states_term):
        states_tensor = torch.tensor(states_term)
        state_XX = []
        for state in states_tensor:
            state = (
                state[: torch.where(state == self.padding_idx)[0][0]]
                if state[-1] == self.padding_idx
                else state
            )
            state_XX.append(state)
        return state_XX

    def statebatch2state(
        self, states: List[TensorType["1", "state_dim"]]
    ) -> TensorType["batch", "state_dim"]:
        if self.tokenizer is not None:
            states = torch.vstack(states)
            states = self.tokenizer.transform(states)
        if isinstance(states, List) and isinstance(states[0], TensorType):
            return torch.stack(states).to(self.device)
        if isinstance(states, List) and isinstance(states[0], List):
            states = torch.tensor(states, dtype=self.float, device=self.device)
            return states
        
    def statetorch2state(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "state_dim"]:
        if self.tokenizer is not None:
            states = self.tokenizer.transform(states)
        return states.to(self.device)

    def plot_reward_distribution(
        self, states=None, scores=None, ax=None, title=None, oracle=None, **kwargs
    ):
        if ax is None:
            fig, ax = plt.subplots()
            standalone = True
        else:
            standalone = False
        if oracle is None:
            oracle = self.oracle
        if title == None:
            title = "Rewards of Sampled States"
        if scores is None:
            if states is None or len(states) == 0:
                return None
            oracle_states = self.statebatch2oracle(states)
            scores = oracle(oracle_states)
        if isinstance(scores, TensorType):
            scores = scores.cpu().detach().numpy()
        ax.hist(scores)
        ax.set_title(title)
        ax.set_ylabel("Number of Samples")
        ax.set_xlabel("Energy")
        if "MES" in title:
            ax.set_xbound(-0.0, 0.01)
        plt.show()
        if standalone == True:
            plt.tight_layout()
            plt.close()
        return ax
    
    def process_dataset(self,item):
        try:
            lst = ast.literal_eval(item)
            return lst
        except (ValueError, SyntaxError):
            return item
        
    def generate_fidelities(self, n_samples, config):
        if config.oracle_dataset.fid_type == "random":
            fidelities = torch.randint(low=0, high=self.n_fid, size=(n_samples, 1)).to(
                self.float
            )
        else:
            raise NotImplementedError
        return fidelities    
    
    def call_oracle_per_fidelity(self, state_oracle, fidelities):
        if fidelities.ndim == 2:
            fidelities = fidelities.squeeze(-1)
        scores = torch.zeros(
            (fidelities.shape[0]), dtype=self.float, device=self.device
        )
        
        for fid in range(self.n_fid):
            idx_fid = torch.where(fidelities == self.oracle[fid].fid)[0]
            if isinstance(state_oracle, torch.Tensor):
                states = state_oracle[idx_fid]
                states = states.to(self.oracle[fid].device)
            else:
                chosen_state_index = torch.zeros(
                    scores.shape, dtype=self.float, device=self.device
                )
                chosen_state_index[idx_fid] = 1
                states = list(
                    itertools.compress(state_oracle, chosen_state_index.tolist())
                )
            if len(states) > 0:
                scores[idx_fid] = self.oracle[fid](states)
        return scores
    
    def write_samples_to_file(self, samples, path):
        samples = [self.state2readable(state) for state in samples]
        df = pd.DataFrame(samples, columns=["samples"])
        df.to_csv(path)
