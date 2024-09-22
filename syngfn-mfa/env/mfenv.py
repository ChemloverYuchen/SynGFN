from abc import abstractmethod
from typing import List, Tuple
import numpy.typing as npt
from torchtyping import TensorType
import numpy as np
import torch
import matplotlib.pyplot as plt
import itertools
import copy
import pandas as pd
import time
import ast
from copy import deepcopy
from gflownet.gflownet.envs.building_block import BuildingBlock as GflowNetBuildingBlock


class MultiFidelityEnvWrapper(GflowNetBuildingBlock):
    def __init__(
        self,
        env,
        n_fid,
        oracle,
        proxy_state_format=None,
        beta_factor=0,
        norm_factor=1,
        proxy=None,
        fid_embed="one_hot",
        fid_embed_dim=None,
        **kwargs
    ):
        self.fid_embed = fid_embed
        if self.fid_embed != "one_hot" and fid_embed_dim is None:
            raise ValueError("fid_embed_dim cannot be None for non-OHE embedding")
        self.fid_embed_dim = fid_embed_dim
        super().__init__(**kwargs)
        self.env = env
        self.is_state_list = True
        if isinstance(self.env.source, TensorType):
            self.is_state_list = False
        self.n_fid = n_fid
        self.fid = self.env.eos_1 + 1
        self.padding_idx = self.env.padding_idx
        vars(self).update(vars(self.env))
        self.reset()
        self.action_space_1 = self.get_actions_space_1()
        
        
        self.fixed_policy_output_1 = np.ones(len(self.action_space_1))
        self.policy_input_dim_1 = len(self.state2policy_1())
        self.policy_output_dim_1 = len(self.fixed_policy_output_1)
        self.random_policy_output_1 = self.fixed_policy_output_1
        
        if proxy_state_format == "oracle":
            self.statebatch2proxy = self.statebatch2oracle
            self.statetorch2proxy = self.statetorch2oracle
        elif proxy_state_format == "state":
            self.statebatch2proxy = self.statebatch2state
            self.statetorch2proxy = self.statetorch2state
        else:
            raise ValueError("Invalid proxy_state_format")
        
        self.oracle = oracle
        self.fidelity_costs = self.set_fidelity_costs()
        if self.is_state_list:
            self.source = self.env.source + [-1]
        else:
            self.source = torch.cat((self.env.source, torch.tensor([-1])))
        self._test_traj_list = []
        self._test_traj_actions_list = []
        

    def write_samples_to_file(self, samples, path):
        if isinstance(samples, List):
            df = pd.DataFrame(samples, columns=["samples"])
            df.to_csv(path)
        else:
            raise ValueError("Samples should be list")
        
    def set_proxy(self, proxy):
        self.proxy = proxy
        self.proxy_factor = 1.0

    def unpad_function(self, states_term):
        if hasattr(self.env, "unpad_function") == False:
            return states_term
        states_tensor = torch.tensor(states_term)
        states_tensor = states_tensor[:, :-1]
        state_XX = []
        for state in states_tensor:
            state = (
                state[: torch.where(state == self.padding_idx)[0][0]]
                if state[-1] == self.padding_idx
                else state
            )
            state_XX.append(state)
        return state_XX

    def statebatch2state_longFid(self, states):
        state, fid = zip(*[[state[:-1], state[-1]] for state in states])
        states = self.env.statebatch2state(list(state))
        if self.is_state_list:
            fidelities = torch.tensor(
                fid, device=states.device, dtype=states.dtype
            ).unsqueeze(-1)
        else:
            fidelities = torch.vstack(fid).to(states.dtype).to(states.device)
        states = torch.cat([states, fidelities], dim=-1)
        return states

    def statetorch2state_longFid(self, states: TensorType["batch", "state_dim"]):
        state = states[:, :-1]
        fid = states[:, -1].unsqueeze(-1)
        state = self.env.statetorch2state(state)
        states = torch.cat([state, fid], dim=-1)
        return states

    def statebatch2state(self, states: List[TensorType["state_dim"]]):
        state, fid_list = zip(*[[state[:-1], state[-1]] for state in states])
        states = self.env.statebatch2state(list(state))
        if self.is_state_list:
            fidelities = torch.tensor(
                fid_list, device=self.device, dtype=self.float
            ).unsqueeze(-1)
        else:
            fidelities = torch.vstack(fid_list).to(self.dtype).to(self.device)
        updated_fidelities = torch.zeros_like(fidelities)
        for fid in range(self.n_fid):
            idx_fid = torch.where(fidelities == fid)[0]
            if idx_fid.numel() > 0:
                updated_fidelities[idx_fid] = self.oracle[fid].fid
        states = torch.cat([states, updated_fidelities], dim=-1)
        return states

    def statetorch2state(self, states):
        if isinstance(states, TensorType) == False:
            raise ValueError("States should be a tensor")
        else:
            states = states.to(self.float).to(self.device)
        fidelities = states[:, -1]
        states = states[:, :-1]
        states = self.env.statetorch2state(states)
        fidelities = fidelities.to(states.dtype).to(states.device)
        updated_fidelities = torch.zeros_like(fidelities)
        for fid in range(self.n_fid):
            idx_fid = torch.where(fidelities == fid)[0]
            updated_fidelities[idx_fid] = self.oracle[fid].fid
        states = torch.cat([states, updated_fidelities.unsqueeze(-1)], dim=-1)
        return states

    def set_fidelity_costs(self):
        fidelity_costs = {}
        if hasattr(self.oracle[0], "fid") == False:
            print(
                "\nUser-Defined Warning: Oracle1 does not have fid. Re-assigning fid to all oracles. \n"
            )
            # if anyone of the oracles don't have the fidelity parameter (fid), fid is reassigned to all oracles
            for idx in range(self.n_fid):
                setattr(self.oracle[idx], "fid", idx)
        for idx in range(self.n_fid):
            fidelity_costs[self.oracle[idx].fid] = self.oracle[idx].cost
        return fidelity_costs

    def copy(self):
        env_copy = self._copy()
        class_var = self.__dict__.copy()
        class_var.pop("env", None)
        return MultiFidelityEnvWrapper(**class_var, env=env_copy)

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

    def get_actions_space_1(self):
        actions = self.env.get_action_space_1()
        for fid_index in range(self.n_fid):
            actions = actions + [(self.fid, fid_index)]
        self.action_max_length = max(len(action) for action in actions)
        self.action_pad_length = self.action_max_length - len(actions[0])
        return actions
          
    def reset(self, env_id=None, fid: int = -1):
        self.env = self.env._reset(env_id)
        if self.is_state_list:
            self.state = self.env.state + [fid]
        else:
            self.state = torch.cat([self.env.state, torch.tensor([fid])], dim=-1)
        self.fid_done = self.state[-1] != -1
        self.done = self.env.done and self.fid_done
        self.id = self.env.id
        self.n_actions = 0
        return self

    def set_state(self, state, done):
        self.state = state
        self.env.state = state[:-1]
        self.done = done
        if done:
            self.fid_done = done
            self.env.done = done
        else:
            self.fid_done = self.state[-1] != -1
            if self.n_actions == self.get_max_traj_len() - 1:
                # If fid is incomplete then env is complete
                # If fid is complete then env is incomplete
                self.env.done = ~self.fid_done
            else:
                self.env.done = done
        return self

    def get_max_traj_len(self):
        return self.env.get_max_traj_len() + 1
    
    def get_mask_invalid_actions_forward_1(self, state=None, done=None, fid_done=None):
        if state is not None and fid_done is None:
            fid_done = state[-1] != -1
        elif state is None and fid_done is None:
            assert self.fid_done == (self.state[-1] != -1)
            fid_done = self.fid_done
        else:
            print("state, fid_done", state, fid_done)
            raise ValueError(
                "Either just state should be provided. Or both state and fid should be provided. Or nothing should be provided for consistency."
            )
        if state is None:
            if self.is_state_list:
                state = self.state.copy()
            else:
                state = self.state.clone()
                if isinstance(fid_done, torch.Tensor):
                    fid_done = fid_done.item()
        if done is None:
            done = self.done
        if done:
            try:
                assert self.env.done == True
            except:
                print(
                    "done is true but env.done is False: {}, {}".format(
                        done, self.env.done
                    )
                )
                raise Exception
            return [True for _ in range(len(self.action_space_1))]#

        if self.env.done:
            mask = [True for _ in range(len(self.env.action_space_1))] + [False for _ in range(self.n_fid)]
        else:
            mask = self.env.get_mask_invalid_actions_forward_1(state[:-1])
            mask = mask + [True for _ in range(self.n_fid)]
        return mask
    
    def get_mask_invalid_actions_forward_2(self, state = None, done = None, action_1 = None):
        if state is None:
            state = self.state.copy()
        if action_1 and action_1[0] == self.fid:
            mask = [False for _ in range(len(self.env.action_space_2))]
            mask[-2] = True
            return mask
        else:
            mask = self.env._get_mask_invalid_actions_forward_2(state=state[:-1],action_1=action_1)
            return mask
    
    def state2policy_1(self, state: List = None):
        if state is None:
            if self.is_state_list:
                state = self.state.copy()
            else:
                state = self.state.clone()
        state_policy = self.env._state2policy_1(state[:-1])
        fid = state[-1]
        if self.fid_embed == "one_hot":
            fid_policy = np.zeros((self.n_fid), dtype=np.float32)
            if len(state) > 0 and fid != -1:
                fid_policy[fid] = 1
        elif self.fid_embed == "thermometer":
            per_fid_dimension = int(self.fid_embed_dim / self.n_fid)
            fid_policy = np.zeros((per_fid_dimension * self.n_fid), dtype=np.float32)
            if len(state) > 0 and fid != -1:
                fid_policy[0 : (fid + 1) * per_fid_dimension] = 1
        if self.is_state_list == False:
            state_policy = state_policy.squeeze(0).detach().cpu().numpy()
        state_fid_policy = np.concatenate((state_policy, fid_policy), axis=0)
        return state_fid_policy

    def statebatch2policy_1(self, states: List[List]) -> npt.NDArray[np.float32]:
        if not states:
            return np.array(states)
        state_policy, fid_list = zip(*[(state[:-1], state[-1]) for state in states])
        state_policy = self.env.statebatch2policy_1(state_policy)
        if self.fid_embed == "one_hot":
            fid_policy = np.zeros((len(states), self.n_fid), dtype=np.float32)
            fid_array = np.array(fid_list, dtype=np.int32)
            index = np.where(fid_array != -1)[0]
            if index.size:
                fid_policy[index, fid_array[index]] = 1
        elif self.fid_embed == "thermometer":
            per_fid_dimension = int(self.fid_embed_dim / self.n_fid)
            fid_policy = np.zeros(
                (len(states), per_fid_dimension * self.n_fid), dtype=np.float32
            )
            fid_array = np.array(fid_list, dtype=np.int32)
            fid_array = fid_array + 1
            upper_lim = fid_array * per_fid_dimension
            mask = np.arange(fid_policy.shape[1]) < upper_lim[:, np.newaxis]
            fid_policy[mask] = 1
        if self.is_state_list == False:
            state_policy = state_policy.squeeze(1).detach().cpu().numpy()
        state_fid_policy = np.concatenate((state_policy, fid_policy), axis=1)
        return state_fid_policy

    def policy2state(self, state_policy: List) -> List:
        policy_no_fid = state_policy[:, : -self.n_fid]
        state_no_fid = self.env.policy2state(policy_no_fid)
        if self.fid_embed == "one_hot":
            fid = np.argmax(state_policy[:, -self.n_fid :], axis=1)
        elif self.fid_embed == "thermometer":
            per_fid_dimension = int(self.fid_embed_dim / self.n_fid)
            fid = (
                np.argmax(state_policy[:, -self.n_fid * per_fid_dimension :], axis=1)
                / per_fid_dimension
            )
        state = state_no_fid + fid.to_list()
        return state

    def statetorch2policy_1(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "policy_input_dim"]:
        state_policy = states[:, :-1]
        state_policy = self.env._statetorch2policy_1(state_policy)
        state_policy = torch.tensor(state_policy, dtype=self.float, device=self.device)
        if self.fid_embed == "one_hot":
            fid_policy = torch.zeros(
                (states.shape[0], self.n_fid), dtype=self.float, device=self.device
            )

            condition = torch.logical_and(
                states[:, -1] >= 0, states[:, -1] < self.n_fid
            ).unsqueeze(-1)
            index = torch.where(condition)[0]
            if index.size:
                fid_policy[index, states[index, -1].long()] = 1
        elif self.fid_embed == "thermometer":
            per_fid_dimension = int(self.fid_embed_dim / self.n_fid)
            fid_tensor = states[:, -1]
            fid_tensor = fid_tensor + 1
            upper_lim = fid_tensor * per_fid_dimension
            fid_policy = torch.zeros(
                (states.shape[0], per_fid_dimension * self.n_fid),
                dtype=self.float,
                device=self.device,
            )
            mask = torch.arange(
                fid_policy.shape[1], device=self.device, dtype=self.float
            ) < upper_lim.unsqueeze(-1)
            fid_policy[mask] = 1
        state_fid_policy = torch.cat((state_policy, fid_policy), dim=1)
        return state_fid_policy

    def state2readable(self, state=None):
        readable_state = self.env._state2readable(state[:-1])
        if isinstance(state[-1], torch.Tensor):
            fid = str(state[-1].detach().cpu().numpy())
        else:
            fid = str(state[-1])
        return readable_state + ";" + fid

    def statetorch2readable(self, state=None, **kwargs):
        try:
            assert torch.eq(state.to(torch.long), state).all()
        except:
            print(state)
            raise Exception("state is not integer")
        state = state.to(torch.long)
        readable_state = self.env.statetorch2readable(state[:-1], **kwargs)
        fid = str(state[-1].detach().cpu().numpy())
        return readable_state + ";" + fid

    def readable2state(self, readable):
        fid = readable.split(";")[-1]
        readable_state = ";".join(readable.split(";")[:-1])
        state = self.env.readable2state(readable_state)
        if self.is_state_list:
            state = state + [int(fid)]
        else:
            state = torch.cat(
                [state, torch.tensor([int(fid)], device=state.device)], dim=-1
            )
        return state
    
    def get_parents(self, state=None, done=None, action=None):
        if self.is_state_list:
            try:
                assert self.state[:-1] == self.env.state
            except:
                print(self.state[:-1], self.env.state)
                raise Exception("state and env.state is not equal in get_parents")
        else:
            try:
                assert torch.eq(self.state[:-1], self.env.state).all()
            except:
                print(self.state[:-1], self.env.state)
                raise Exception("state and env.state is not equal in get_parents")
        if state is None:
            if self.is_state_list:
                state = self.state.copy()
            else:
                state = self.state.clone()
        if done is None:
            done = self.done
        if (self.is_state_list and state == self.source and self.env.done == False) or (
            self.is_state_list == False
            and torch.eq(state, self.source).all()
            and self.env.done == False
        ):
            
            return [], []
        
        parents_no_fid, actions_1, actions_2 = self.env.get_parents(state[:-1])


        if state[-1] == -1:
            if self.is_state_list:
                parents = [parent + [-1] for parent in parents_no_fid]
            else:
                parents_tensor = torch.vstack(parents_no_fid)
                fid_tensor = torch.ones((parents_tensor.shape[0], 1)) * -1
                parents = torch.cat([parents_tensor, fid_tensor], dim=-1)
                parents = list(parents)
            """
            If fidelity has been chosen, then possible parents include:
            1. The parent state with fidelity set to state's fidelity
            2. The parent with action as choosing that fidelity
            """
        elif state[-1] != -1:
            if self.is_state_list:
                fid = state[-1]
                parents = [parent + [fid] for parent in parents_no_fid]
                fid_parent = state[:-1] + [-1]
            else:
                fid = state[-1].item()
                fid_tensor = torch.tensor([fid], device=state.device)
                parents = [
                    torch.cat([parent, fid_tensor], dim=-1) for parent in parents_no_fid
                ]
                fid_parent = torch.cat([state[:-1], torch.tensor([-1])], dim=-1)
            actions_1.append(tuple([self.fid, fid] + [0] * (self.action_max_length - 2)))
            parents.append(fid_parent)
        return parents, actions_1, actions_2

    def step(self, action_1, action_2):
        if self.is_state_list:
            try:
                assert self.state[:-1] == self.env.state
            except:
                print(self.state[:-1], self.env.state)
                raise Exception("state and env.state is not equal in step")
        else:
            try:
                assert torch.eq(self.state[:-1], self.env.state).all()
            except:
                print(self.state[:-1], self.env.state)
                raise Exception("state and env.state is not equal in step")
        try:
            assert self.done == (self.env.done and self.fid_done)
        except:
            print(self.done, self.env.done, self.fid_done)
            raise Exception("done is not equal to (env.done and fid_done) in step")
        if self.state[-1] == -1:
            try:
                assert self.n_actions == self.env.n_actions
            except:
                print(self.n_actions, self.env.n_actions)
                raise Exception(
                    "n_actions is not equal to env.n_actions in step when fid has not been chosen"
                )
        else:
            try:
                assert self.n_actions == self.env.n_actions + 1
            except:
                print(self.n_actions, self.env.n_actions)
                raise Exception(
                    "n_actions is not equal to env.n_actions + 1 in step when fid has been chosen"
                )
        if self.done:
            raise ValueError("Action has been sampled despite environment being done")
        
        if action_1[0] == self.fid:
            if self.fid_done == False:
                self.state[-1] = action_1[1]
                self.fid_done = True
                self.n_actions += 1
            else:
                raise ValueError("Fidelity has already been chosen.")
            assert self.fid_done == (self.state[-1] != -1)
            self.done = self.env.done and self.fid_done
            return self.state, action_1, action_2, True
        else:
            fid = self.state[-1]
            state, action_1, action_2, valid = self.env._step(action_1, action_2)
            if valid:
                self.n_actions += 1
            if self.is_state_list:
                self.state = state + [fid]
            else:
                fid = torch.tensor([fid], device=state.device)
                self.state = torch.cat([state, fid], dim=-1)
            assert self.fid_done == (self.state[-1] != -1)
            if self.fid_done:
                assert self.n_actions == self.env.n_actions + 1
            else:
                assert self.n_actions == self.env.n_actions
            self.done = self.env.done and self.fid_done
            return self.state, action_1, action_2, valid

    def statebatch2oracle(self, states: List[List]):
        states, fid_list = zip(*[(state[:-1], state[-1]) for state in states])
        state_oracle = self.env.statebatch2oracle(list(states))
        fidelities = torch.tensor(
            fid_list, dtype=torch.int64, device=self.device
        ).unsqueeze(-1)
        transformed_fidelities = torch.zeros_like(fidelities)
        for fid in range(self.n_fid):
            idx_fid = torch.where(fidelities == fid)[0]
            if idx_fid.shape[0] > 0:
                transformed_fidelities[idx_fid] = self.oracle[fid].fid
        return state_oracle, transformed_fidelities
    
    def statebatch2smiles(self, states: List[List]):
        states, fid_list = zip(*[(state[:-1], state[-1]) for state in states])
        states = self.env.statebatch2oracle(list(states))
        return states
    
    def statetorch2oracle(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "oracle_dim"]:
        state_oracle = states[:, :-1]
        state_oracle = self.env.statetorch2oracle(state_oracle)
        fidelities = states[:, -1]
        transformed_fidelities = torch.zeros_like(fidelities)
        for fid in range(self.n_fid):
            idx_fid = torch.where(fidelities == fid)[0]
            if idx_fid.shape[0] > 0:
                transformed_fidelities[idx_fid] = self.oracle[fid].fid
        return state_oracle, transformed_fidelities

    def get_uniform_terminating_states(self, n_states: int) -> List[List]:
        if hasattr(self.env, "get_uniform_terminating_states"):
            states = self.env.get_uniform_terminating_states(n_states)
            fidelities = torch.randint(0, self.n_fid, (n_states,)).tolist()
            state_fid = [state + [fid] for state, fid in zip(states, fidelities)]
            return state_fid
        else:
            return None

    def plot_samples_frequency(self, samples, title=None, **kwargs):
        width = (self.n_fid) * 5
        fig, axs = plt.subplots(1, self.n_fid, figsize=(width, 5))
        for fid in range(0, self.n_fid):
            if isinstance(samples, TensorType):
                samples = samples.tolist()
                # samples_fid = samples[:, ]
            # else:
            samples_fid = [sample[:-1] for sample in samples if sample[-1] == fid]
            if hasattr(self.env, "plot_samples_frequency"):
                axs[fid] = self.env.plot_samples_frequency(
                    samples_fid, axs[fid], "Fidelity {}".format(fid)
                )
            else:
                return None
        fig.suptitle(kwargs.get("title", "Frequency of Coordinates Sampled"))
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
        plt.close()
        return fig

    def plot_reward_distribution(
        self, states=None, scores=None, fidelity=None, **kwargs
    ):
        width = (self.n_fid) * 5
        fig, axs = plt.subplots(1, self.n_fid, figsize=(width, 5))
        for fid in range(0, self.n_fid):
            if states is not None:
                samples_fid = [state[:-1] for state in states if state[-1] == fid]
                if hasattr(self.env, "plot_reward_distribution"):
                    axs[fid] = self.env.plot_reward_distribution(
                        states=samples_fid,
                        ax=axs[fid],
                        title="Fidelity {}".format(fid),
                        oracle=self.oracle[fid],
                    )
            elif scores is not None:
                title = kwargs.get("title", None)
                if "Initial Dataset" in title:
                    idx_fid = [i for i in range(len(scores)) if int(fidelity[i]) == fid]
                else:
                    idx_fid = [
                        i
                        for i in range(len(scores))
                        if fidelity[i] == self.oracle[fid].fid
                    ]
                if hasattr(self.env, "plot_reward_distribution") and len(idx_fid) != 0:
                    axs[fid] = self.env.plot_reward_distribution(
                        scores=scores[idx_fid],
                        ax=axs[fid],
                        title="Fidelity {}".format(fid),
                    )
            else:
                return None
        fig.suptitle(kwargs.get("title", "Reward Distribution"))
        plt.tight_layout()
        plt.show()
        plt.close()
        return fig

    def get_cost(self, samples, fidelities=None):
        if fidelities is None:
            fidelities = [self.oracle[int(sample[-1])].fid for sample in samples]
        if isinstance(fidelities, TensorType):
            if fidelities.ndim == 2:
                fidelities = fidelities.squeeze(-1)
            fidelities = fidelities.tolist()
        costs = [self.fidelity_costs[fid] for fid in fidelities]
        return costs

    def get_pairwise_distance(self, samples_set1, samples_set2=None):
        if hasattr(self.env, "get_pairwise_distance"):
            return self.env.get_pairwise_distance(samples_set1, samples_set2)
        else:
            return torch.zeros(len(samples_set1))

    def get_distance_from_D0(self, samples, dataset_obs):
        if hasattr(self.env, "get_distance_from_D0"):
            dataset_obs = [sample[:-1] for sample in dataset_obs]
            return self.env.get_distance_from_D0(samples, dataset_obs)
        else:
            return torch.zeros(len(samples))

    def generate_fidelities(self, n_samples, config):
        if config.oracle_dataset.fid_type == "random":
            fidelities = torch.randint(low=0, high=self.n_fid, size=(n_samples, 1)).to(
                self.float
            )
        else:
            raise NotImplementedError
        return fidelities

    def process_dataset(self,item):
        try:
            lst = ast.literal_eval(item)
            return lst
        except (ValueError, SyntaxError):
            return item
    
    def initialize_dataset(self, config, n_samples, resume):
        train_scores = []
        test_scores = []
        train_samples = []
        test_samples = []
        
        if config.oracle_dataset is not None:
            # Train Path Given
            if config.oracle_dataset.train is not None:
                train = pd.read_csv(config.oracle_dataset.train.path)
                train["samples"] = train["samples"].apply(self.process_dataset)
                train_samples = train["samples"].values.tolist()
                if config.oracle_dataset.train.get_scores == False:
                    train_scores = train["energies"].values.tolist()
            # Test Path Given
            if config.oracle_dataset.test is not None:
                test = pd.read_csv(config.oracle_dataset.test.path)
                test["samples"] = test["samples"].apply(self.process_dataset)
                test_samples = test["samples"].values.tolist()
                if config.oracle_dataset.test.get_scores == False:
                    test_scores = test["energies"].values.tolist()

        # If neither were given, generate dataset
        if train_samples == [] and test_samples == []:
            raise ValueError("No dataset provided")
        else:
            #train_states:traj
            train_states = [
                torch.tensor(sample)
                for sample in train_samples
            ]
            test_states = [
                torch.tensor(sample)
                for sample in test_samples
            ]
            states = train_states + test_states

            if train_states != []:
                train_states = torch.stack(train_states)
            if test_states != []:
                test_states = torch.stack(test_states)
            
            states = torch.stack(states)
            
        scores = train_scores + test_scores
        if scores == []:
            train_states_oracle_input = train_states.clone()
            train_state_oracle, fid = self.statetorch2oracle(train_states_oracle_input)
            train_scores = self.call_oracle_per_fidelity(train_state_oracle, fid)

            if test_states != []:
                test_states_oracle_input = test_states.clone()
                test_state_oracle, fid = self.statetorch2oracle(
                    test_states_oracle_input
                )
                test_scores = self.call_oracle_per_fidelity(test_state_oracle, fid)
            else:
                test_scores = torch.tensor([], dtype=self.float, device=self.device)
                test_states = torch.tensor([], dtype=self.float, device=self.device)

            scores = torch.cat([train_scores, test_scores], dim=0)

        if isinstance(scores, List):
            scores = torch.tensor(scores, dtype=self.float)
        if resume == False:
            return states, scores
        else:
            if isinstance(train_scores, List):
                train_scores = torch.tensor(train_scores, dtype=self.float)
            if isinstance(test_scores, List):
                test_scores = torch.tensor(test_scores, dtype=self.float)
            return train_states, train_scores, test_states, test_scores