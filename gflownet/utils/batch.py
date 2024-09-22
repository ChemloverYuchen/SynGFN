from __future__ import annotations
from collections import OrderedDict
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import numpy.typing as npt
import torch
from torchtyping import TensorType

from gflownet.envs.building_block import BuildingBlock
from gflownet.utils.common import (
    concat_items,
    copy,
    extend,
    set_device,
    set_float_precision,
    tbool,
    tfloat,
    tlong,
)

class Batch:
    def __init__(
        self,
        env: Optional[BuildingBlock] = None,
        device: Union[str, torch.device] = "cuda",
        float_type: Union[int, torch.dtype] = 32,
    ):
        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_type)
        # Generic environment, properties and dictionary of state and forward mask of
        # source (as tensor)
        if env is not None:
            self.set_env(env)
        else:
            self.env = None
            self.source = None
        # Initialize batch size 0
        self.size = 0
        # Initialize empty batch variables
        self.envs: OrderedDict[int, BuildingBlock] = OrderedDict()# {env.id:env}
        self.trajectories = OrderedDict()# {env.id:[batch_indices]}
        self.is_backward = OrderedDict()
        self.traj_indices = []# [env.id,env.id']
        self.state_indices = []
        self.states = []
        self.actions_1 = []
        self.actions_2 = []
        self.done = []
        self.masks_invalid_actions_forward_1 = []
        self.masks_invalid_actions_forward_2 = []
        self.masks_invalid_actions_backward_1 = []
        self.masks_invalid_actions_backward_2 = []
        
        self.parents = []
        self.parents_all = []
        self.parents_actions_all_1 = []
        self.parents_actions_all_2 = []
        self.n_actions = []
        self.states_policy_1 = None
        self.states_policy_2 = None
        self.parents_policy_1 = None
        self.parents_policy_2 = None
        
        # Flags for available items
        self.parents_available = True
        self.parents_policy_available_1 = False
        self.parents_policy_available_2 = False
        self.parents_all_available = False
        self.masks_forward_available_1 = False
        self.masks_forward_available_2 = False
        self.masks_backward_available_1 = False
        self.masks_backward_available_2 = False
        self.rewards_available = False

    #batch size
    def __len__(self):
        return self.size
    
    def get_donelist(self):
        return self.done
    
    def batch_idx_to_traj_state_idx(self, batch_idx: int):
        traj_idx = self.traj_indices[batch_idx]
        state_idx = self.state_indices[batch_idx]
        return traj_idx, state_idx

    def traj_idx_to_batch_indices(self, traj_idx: int):
        batch_indices = self.trajectories[traj_idx]
        return batch_indices

    def traj_state_idx_to_batch_idx(self, traj_idx: int, state_idx: int):
        batch_idx = self.trajectories[traj_idx][state_idx]
        return batch_idx
    
    
    def traj_idx_action_idx_to_batch_idx(
        self, traj_idx: int, action_idx: int, backward: bool
    ):
        if traj_idx not in self.trajectories:
            return None
        if backward:
            if action_idx >= len(self.trajectories[traj_idx]):
                return None
            return self.trajectories[traj_idx][::-1][action_idx]
        if action_idx > len(self.trajectories[traj_idx]):
            return None
        return self.trajectories[traj_idx][action_idx - 1]
   
    def idx2state_idx(self, idx: int):
        return self.trajectories[self.traj_indices[idx]].index(idx)

    
    def set_env(self, env: BuildingBlock):
        self.env = env.copy().reset()
        self.source = {
            "state": self.env.source,
            "mask_forward_1": tbool(
                self.env.get_mask_invalid_actions_forward_1(), device=self.device
            ),
            "mask_forward_2": tbool(
                self.env.get_mask_invalid_actions_forward_2(), device=self.device
            ),
        }


    def add_to_batch(
        self,
        envs: List[BuildingBlock],
        actions_1: List[Tuple],
        actions_2: List[Tuple],
        valids: List[bool],
        backward: Optional[bool] = False,
        train: Optional[bool] = True,
    ):
        # Add data samples to the batch
        for env, action_1, action_2, valid in zip(envs, actions_1, actions_2, valids):
            if train is False and env.done is False:
                continue
            # Add env to dictionary
            if env.id not in self.envs:
                self.envs.update({env.id: env})
            # Add batch index to trajectory
            if env.id not in self.trajectories:
                self.trajectories.update({env.id: [len(self)]})
            else:
                if backward:
                    self.trajectories[env.id].insert(0, len(self))
                else:
                    self.trajectories[env.id].append(len(self))
            # Set whether trajectory is backward
            if env.id not in self.is_backward:
                self.is_backward.update({env.id: backward})
            
            if backward:
                if env.state != self.source["state"]:
                    self.traj_indices.append(env.id)
                    self.state_indices.append(env.n_actions)
                    self.states.append(copy(env.state))
                    # Set masks to None
                    self.masks_invalid_actions_forward_1.append(None)
                    self.masks_invalid_actions_forward_2.append(None)
                    self.masks_invalid_actions_backward_1.append(None)
                    self.masks_invalid_actions_backward_2.append(None)
                    # Increment size of batch
                    self.size += 1
                if len(self.trajectories[env.id]) == 1:
                    self.done.append(True)
                else:
                    self.actions_1.append(action_1)
                    self.actions_2.append(action_2)
                    if env.state != self.source["state"]:
                        self.parents.append(copy(self.states[self.trajectories[env.id][0]]))
                        self.done.append(False)
                    else:
                        self.parents.append(copy(self.source["state"]))
                        self.trajectories[env.id] = self.trajectories[env.id][1:] 
                
            else:
                self.traj_indices.append(env.id)
                self.state_indices.append(env.n_actions)
                self.actions_1.append(action_1)
                self.actions_2.append(action_2)
                self.states.append(copy(env.state))
                self.done.append(env.done)
                if len(self.trajectories[env.id]) == 1:
                    self.parents.append(copy(self.source["state"]))
                else:
                    self.parents.append(
                        copy(self.states[self.trajectories[env.id][-2]])
                    )
        
                # Set masks to None
                self.masks_invalid_actions_forward_1.append(None)
                self.masks_invalid_actions_backward_1.append(None)
                self.masks_invalid_actions_forward_2.append(None)
                self.masks_invalid_actions_backward_2.append(None)
                # Increment size of batch
                self.size += 1
                
        # Other variables are not available after new items were added to the batch
        self.masks_forward_available_1 = False
        self.masks_forward_available_2 = False
        self.masks_backward_available_1 = False
        self.masks_backward_available_2 = False
        self.parents_policy_available_1 = False
        self.parents_policy_available_2 = False
        self.parents_all_available = False
        self.rewards_available = False

    def get_n_trajectories(self) -> int:
        return len(self.trajectories)

    def get_unique_trajectory_indices(self) -> List:
        return list(self.trajectories.keys())

    def get_trajectory_indices(
        self, consecutive: bool = False, return_mapping_dict: bool = False
    ) -> TensorType["n_states", int]:
        if consecutive:
            traj_index_to_consecutive_dict = {
                traj_idx: consecutive
                for consecutive, traj_idx in enumerate(self.trajectories)
            }
            traj_indices = list(
                map(lambda x: traj_index_to_consecutive_dict[x], self.traj_indices)
            )
        else:
            traj_indices = self.traj_indices
        if return_mapping_dict and consecutive:
            return (
                tlong(traj_indices, device=self.device),
                traj_index_to_consecutive_dict,
            )
        else:
            return tlong(traj_indices, device=self.device)

    def get_state_indices(self) -> TensorType["n_states", int]:
        return tlong(self.state_indices, device=self.device)

    def get_states_1(
        self,
        policy: Optional[bool] = False,
        proxy: Optional[bool] = False,
        force_recompute: Optional[bool] = False,
    ) -> Union[TensorType["n_states", "..."], npt.NDArray[np.float32], List]:
        if policy is True and proxy is True:
            raise ValueError(
                "Ambiguous request! Only one of policy or proxy can be True."
            )
        if policy is True:
            if self.states_policy_1 is None or force_recompute is True:
                self.states_policy_1 = self.states2policy_1()
            return self.states_policy_1
        if proxy is True:
            return self.states2proxy()
        return self.states

    def get_states_2(
        self,
        policy: Optional[bool] = False,
        proxy: Optional[bool] = False,
        force_recompute: Optional[bool] = False,
    ) -> Union[TensorType["n_states", "..."], npt.NDArray[np.float32], List]:
        if policy is True and proxy is True:
            raise ValueError(
                "Ambiguous request! Only one of policy or proxy can be True."
            )
        if policy is True:
            if self.states_policy_2 is None or force_recompute is True:
                self.states_policy_2 = self.states2policy_2()
            return self.states_policy_2
        if proxy is True:
            return self.states2proxy()
        return self.states
    
    def states2policy_1(
        self,
        states: Optional[Union[List[List], List[TensorType["n_states", "..."]]]] = None,
        traj_indices: Optional[Union[List, TensorType["n_states"]]] = None,
    ) -> TensorType["n_states", "state_policy_dims"]:
        if states is None:
            states = self.states
        return tfloat(
            self.env.statebatch2policy_1(states),
            device=self.device,
            float_type=self.float,
        )

    def states2policy_2(
        self,
        states: Optional[Union[List[List], List[TensorType["n_states", "..."]]]] = None,
        traj_indices: Optional[Union[List, TensorType["n_states"]]] = None,
        actions: List[Tuple] = None,
    ) -> TensorType["n_states", "state_policy_dims"]:
        if states is None:
            states = self.parents
            actions = self.actions_1
        return tfloat(
            self.env.statebatch2policy_2(states,actions),
            device=self.device,
            float_type=self.float,
        )
    
    def states2proxy(
        self,
        states: Optional[Union[List[List], List[TensorType["n_states", "..."]]]] = None,
        traj_indices: Optional[Union[List, TensorType["n_states"]]] = None,
    ) -> Union[
        TensorType["n_states", "state_proxy_dims"], npt.NDArray[np.float32], List
    ]:
        if states is None:
            states = self.states
        return self.env.statebatch2proxy(states)

    def get_actions_1(self) -> TensorType["n_states, action_dim_1"]:
        return tfloat(self.actions_1, float_type=self.float, device=self.device)
    
    def get_actions_2(self) -> TensorType["n_states, action_dim_2"]:
        return tfloat(self.actions_2, float_type=self.float, device=self.device)

    def get_done(self) -> TensorType["n_states"]:
        return tbool(self.done, device=self.device)

    def get_parents_1(
        self, policy: Optional[bool] = False, force_recompute: Optional[bool] = False
    ) -> TensorType["n_states", "..."]:
        if self.parents_available is False or force_recompute is True:
            self._compute_parents()
        if policy:
            if self.parents_policy_available_1 is False or force_recompute is True:
                self._compute_parents_policy_1()
            return self.parents_policy_1
        else:
            return self.parents

    def get_parents_2(
        self, policy: Optional[bool] = False, force_recompute: Optional[bool] = False
    ) -> TensorType["n_states", "..."]:
        if self.parents_available is False or force_recompute is True:
            self._compute_parents()
        if policy:
            if self.parents_policy_available_2 is False or force_recompute is True:
                self._compute_parents_policy_2()
            
            return self.parents_policy_2
        else:
            return self.parents
        
    def _compute_parents(self):
        self.parents = []
        indices = []
        # Iterate over the trajectories to obtain the parents from the states
        for traj_idx, batch_indices in self.trajectories.items():
            # parent is source
            self.parents.append(self.envs[traj_idx].source)
            # parent is not source
            self.parents.extend([self.states[idx] for idx in batch_indices[:-1]])
            indices.extend(batch_indices)
        # Sort parents list in the same order as states
        self.parents = [self.parents[indices.index(idx)] for idx in range(len(self))]
        self.parents_available = True

    def _compute_parents_policy_1(self):
        self.states_policy_1 = self.get_states_1(policy=True)
        self.parents_policy_1 = torch.zeros_like(self.states_policy_1)
        # Iterate over the trajectories to obtain the parents from the states
        for traj_idx, batch_indices in self.trajectories.items():
            # parent is source
            self.parents_policy_1[batch_indices[0]] = tfloat(
                self.envs[traj_idx].state2policy_1(self.envs[traj_idx].source),
                device=self.device,
                float_type=self.float,
            )
            # parent is not source
            self.parents_policy_1[batch_indices[1:]] = self.states_policy_1[
                batch_indices[:-1]
            ]
        self.parents_policy_available_1 = True

    def _compute_parents_policy_2(self):
        self.parents_policy_2 = self.get_states_2(policy=True)
    
    def get_parents_all(
        self, policy: bool = False, force_recompute: bool = False
    ) -> Tuple[
        Union[List, TensorType["n_parents", "..."]],
        TensorType["n_parents", "..."],
        TensorType["n_parents"],
    ]:
        if self.parents_all_available is False or force_recompute is True:
            self._compute_parents_all()
        if policy:
            return (
                self.parents_all_policy_1,
                self.parents_all_policy_2,
                self.parents_actions_all_1,
                self.parents_actions_all_2,
                self.parents_all_indices,
            )
        else:
            return self.parents_all, self.parents_actions_all_1, self.parents_actions_all_2, self.parents_all_indices

    def _compute_parents_all(self):
        # Iterate over the trajectories to obtain all parents
        self.parents_all = []
        self.parents_actions_all_1 = []
        self.parents_actions_all_2 = []
        self.parents_all_indices = []
        self.parents_all_policy_1 = []
        self.parents_all_policy_2 = []
        for idx, traj_idx in enumerate(self.traj_indices):
            state = self.states[idx]
            done = self.done[idx]
            action = self.actions[idx]
            parents, parents_a_1, parents_a_2 = self.envs[traj_idx].get_parents(
                state=state,
                done=done,
                action=action,
            )
            self.parents_all.extend(parents)
            self.parents_actions_all_1.extend(parents_a_1)
            self.parents_actions_all_1.extend(parents_a_2)
            self.parents_all_indices.extend([idx] * len(parents))
            self.parents_all_policy_1.append(
                tfloat(
                    self.envs[traj_idx].statebatch2policy_1(parents),
                    device=self.device,
                    float_type=self.float,
                )
            )
            self.parents_all_policy_2.append(
                tfloat(
                    self.envs[traj_idx].statebatch2policy_2(parents),
                    device=self.device,
                    float_type=self.float,
                )
            )
        # Convert to tensors
        self.parents_actions_all_1 = tfloat(
            self.parents_actions_all_1,
            device=self.device,
            float_type=self.float,
        )
        self.parents_actions_all_2 = tfloat(
            self.parents_actions_all_2,
            device=self.device,
            float_type=self.float,
        )
        self.parents_all_indices = tlong(
            self.parents_all_indices,
            device=self.device,
        )
        self.parents_all_policy_1 = torch.cat(self.parents_all_policy_1)
        self.parents_all_policy_2 = torch.cat(self.parents_all_policy_2)
        self.parents_all_available = True

    def get_masks_forward(
        self,
        of_parents: bool = False,
        force_recompute: bool = False,
    ) -> TensorType["n_states", "action_space_dim"]:
        if self.masks_forward_available_1 is False or force_recompute is True:
            self._compute_masks_forward_1()
        if self.masks_forward_available_2 is False or force_recompute is True:
            self._compute_masks_forward_2()
            
        # Make tensor
        masks_invalid_actions_forward_1 = tbool(
            self.masks_invalid_actions_forward_1, device=self.device
        )   
        masks_invalid_actions_forward_2 = tbool(
            self.masks_invalid_actions_forward_2, device=self.device
        )
        if of_parents:
            trajectories_parents = {
                traj_idx: [-1] + batch_indices[:-1]
                for traj_idx, batch_indices in self.trajectories.items()
            }
            parents_indices = tlong(
                [
                    trajectories_parents[traj_idx][
                        self.trajectories[traj_idx].index(idx)
                    ]
                    for idx, traj_idx in enumerate(self.traj_indices)
                ],
                device=self.device,
            )
            
            masks_invalid_actions_forward_parents_1 = torch.zeros_like(
                masks_invalid_actions_forward_1
            )
            masks_invalid_actions_forward_parents_2 = torch.zeros_like(
                masks_invalid_actions_forward_2
            )
            
            masks_invalid_actions_forward_parents_1[parents_indices == -1] = self.source[
                "mask_forward_1"
            ]
            masks_invalid_actions_forward_parents_2[parents_indices == -1] = self.source[
                "mask_forward_2"
            ]
            
            masks_invalid_actions_forward_parents_1[
                parents_indices != -1
            ] = masks_invalid_actions_forward_1[parents_indices[parents_indices != -1]]
            
            masks_invalid_actions_forward_parents_2[
                parents_indices != -1
            ] = masks_invalid_actions_forward_2[parents_indices[parents_indices != -1]]
            
            return masks_invalid_actions_forward_parents_1, masks_invalid_actions_forward_parents_2
        
        return masks_invalid_actions_forward_1, masks_invalid_actions_forward_2

    def _compute_masks_forward_1(self):
        for idx, mask in enumerate(self.masks_invalid_actions_forward_1):
            if mask is not None:
                continue
            state = self.states[idx]#final state
            done = self.done[idx]
            traj_idx = self.traj_indices[idx]
            self.masks_invalid_actions_forward_1[idx] = self.envs[
                traj_idx
            ].get_mask_invalid_actions_forward_1(state, done)
        self.masks_forward_available_1 = True

    def _compute_masks_forward_2(self):
        for idx, mask in enumerate(self.masks_invalid_actions_forward_2):
            if mask is not None:
                continue
            state = self.states[idx]
            done = self.done[idx]
            traj_idx = self.traj_indices[idx]
            if not done:
                new_idx = (self.trajectories[traj_idx].index(idx) + 1) % len(self.trajectories[traj_idx])
                new_idx = self.trajectories[traj_idx][new_idx]
            else:
                new_idx = idx
            done = self.done[new_idx]
            action_1 = self.actions_1[new_idx]
            self.masks_invalid_actions_forward_2[idx] = self.envs[
                traj_idx
            ].get_mask_invalid_actions_forward_2(state, done, action_1)        
        self.masks_forward_available_2 = True
        
    def get_masks_backward(
        self,
        force_recompute: bool = False,
    ) -> TensorType["n_states", "action_space_dim"]:
        if self.masks_backward_available_1 is False or force_recompute is True:
            self._compute_masks_backward_1()
        if self.masks_backward_available_2 is False or force_recompute is True:
            self._compute_masks_backward_2()
            
        return tbool(self.masks_invalid_actions_backward_1, device=self.device), tbool(self.masks_invalid_actions_backward_2, device=self.device)

    def _compute_masks_backward_1(self):
        # Iterate over the trajectories to compute all backward masks
        for idx, mask in enumerate(self.masks_invalid_actions_backward_1):
            if mask is not None:
                continue
            state = self.states[idx]
            done = self.done[idx]
            traj_idx = self.traj_indices[idx]
            self.masks_invalid_actions_backward_1[idx] = self.envs[
                traj_idx
            ].get_mask_invalid_actions_backward_1(state, done)
        self.masks_backward_available_1 = True
    
    def _compute_masks_backward_2(self): 
        for idx, mask in enumerate(self.masks_invalid_actions_backward_2):
            if mask is not None:
                continue
            state = self.states[idx]
            done = self.done[idx]
            traj_idx = self.traj_indices[idx]
            self.masks_invalid_actions_backward_2[idx] = self.envs[
                traj_idx
            ].get_mask_invalid_actions_backward_2(state, done)
        self.masks_backward_available_2 = True

    def get_rewards(
        self, force_recompute: Optional[bool] = False
    ) -> TensorType["n_states"]:
        if self.rewards_available is False or force_recompute is True:
            self._compute_rewards()
        return self.rewards

    def _compute_rewards(self):
        states_proxy_done = self.get_terminating_states(proxy=True)
        self.rewards = torch.zeros(len(self), dtype=self.float, device=self.device)
        done = self.get_done()
        if len(done) > 0:
            self.rewards[done] = self.env.proxy2reward(
                self.env.proxy(states_proxy_done)
            )
        self.rewards_available = True

    def get_terminating_states(
        self,
        sort_by: str = "insertion",
        policy_1: Optional[bool] = False,
        policy_2: Optional[bool] = False,
        proxy: Optional[bool] = False,
    ) -> Union[TensorType["n_trajectories", "..."], npt.NDArray[np.float32], List]:
        if sort_by == "insert" or sort_by == "insertion":
            indices = np.arange(len(self))
        elif sort_by == "traj" or sort_by == "trajectory":
            indices = np.argsort(self.traj_indices)
        else:
            raise ValueError("sort_by must be either insert[ion] or traj[ectory]")
        if (policy_1 and proxy) or (policy_2 and proxy):
            raise ValueError(
                "Ambiguous request! Only one of policy or proxy can be True."
            )
        traj_indices = None
        if torch.is_tensor(self.states):
            indices = tlong(indices, device=self.device)
            done = self.get_done()[indices]
            states_term = self.states[indices][done, :]
        elif isinstance(self.states, list):
            states_term = [self.states[idx] for idx in indices if self.done[idx]]
            actions_term = [self.actions_1[idx] for idx in indices if self.done[idx]]
        else:
            raise NotImplementedError("self.states can only be list or torch.tensor")
        if policy_1 is True:
            return self.states2policy_1(states_term, traj_indices)
        elif policy_2 is True:
            return self.states2policy_2(states_term, traj_indices, actions_term)
        elif proxy is True:
            return self.states2proxy(states_term, traj_indices)
        else:
            return states_term

    def get_terminating_rewards(
        self,
        sort_by: str = "insertion",
        force_recompute: Optional[bool] = True,
    ) -> TensorType["n_trajectories"]:
        if sort_by == "insert" or sort_by == "insertion":
            indices = np.arange(len(self))
        elif sort_by == "traj" or sort_by == "trajectory":
            indices = np.argsort(self.traj_indices)
        else:
            raise ValueError("sort_by must be either insert[ion] or traj[ectory]")
        if self.rewards_available is False or force_recompute is True:
            self._compute_rewards()
        done = self.get_done()[indices]
        return self.rewards[indices][done]

    def get_actions_trajectories_1(self) -> List[List[Tuple]]:
        actions_trajectories = []
        for batch_indices in self.trajectories.values():
            actions_trajectories.append([self.actions_1[idx] for idx in batch_indices])
        return actions_trajectories
    
    def get_actions_trajectories_2(self) -> List[List[Tuple]]:
        actions_trajectories = []
        for batch_indices in self.trajectories.values():
            actions_trajectories.append([self.actions_2[idx] for idx in batch_indices])
        return actions_trajectories

    def get_states_of_trajectory(
        self,
        traj_idx: int,
        states: Optional[
            Union[TensorType["n_states", "..."], npt.NDArray[np.float32], List]
        ] = None,
        traj_indices: Optional[Union[List, TensorType["n_states"]]] = None,
    ) -> Union[
        TensorType["n_states", "state_proxy_dims"], npt.NDArray[np.float32], List
    ]:
        # If either states or traj_indices are not None, both must be the same type and
        # have the same length.
        if states is not None or traj_indices is not None:
            assert type(states) == type(traj_indices)
            assert len(states) == len(traj_indices)
        else:
            states = self.states
            traj_indices = self.traj_indices
        if torch.is_tensor(states):
            return states[tlong(traj_indices, device=self.device) == traj_idx]
        elif isinstance(states, list):
            return [
                state for state, idx in zip(states, traj_indices) if idx == traj_idx
            ]
        elif isinstance(states, np.ndarray):
            return states[np.array(traj_indices) == traj_idx]
        else:
            raise ValueError("states can only be list, torch.tensor or ndarray")

    def merge(self, batches: List['Batch']):
        if not isinstance(batches, list):
            batches = [batches]
        for batch in batches:
            if len(batch) == 0:
                continue
            # Shift trajectory indices of batch to merge
            if len(self) == 0:
                traj_idx_shift = 0
            else:
                traj_idx_shift = np.max(list(self.trajectories.keys())) + 1
            batch._shift_indices(traj_shift=traj_idx_shift, batch_shift=len(self))
            # Merge main data
            self.size += batch.size
            self.envs.update(batch.envs)
            self.trajectories.update(batch.trajectories)
            self.traj_indices.extend(batch.traj_indices)
            self.state_indices.extend(batch.state_indices)
            self.states.extend(batch.states)
            self.actions_1.extend(batch.actions_1)
            self.actions_2.extend(batch.actions_2)
            self.done.extend(batch.done)
            self.masks_invalid_actions_forward_1 = extend(
                self.masks_invalid_actions_forward_1,
                batch.masks_invalid_actions_forward_1,
            )
            self.masks_invalid_actions_forward_2 = extend(
                self.masks_invalid_actions_forward_2,
                batch.masks_invalid_actions_forward_2,
            )
            self.masks_invalid_actions_backward_1 = extend(
                self.masks_invalid_actions_backward_1,
                batch.masks_invalid_actions_backward_1,
            )
            self.masks_invalid_actions_backward_2 = extend(
                self.masks_invalid_actions_backward_2,
                batch.masks_invalid_actions_backward_2,
            )
            
            # Merge "optional" data
            if self.states_policy_1 is not None and batch.states_policy_1 is not None:
                self.states_policy_1 = extend(self.states_policy_1, batch.states_policy_1)
            else:
                self.states_policy_1 = None
            if self.states_policy_2 is not None and batch.states_policy_2 is not None:
                self.states_policy_2 = extend(self.states_policy_2, batch.states_policy_2)
            else:
                self.states_policy_2 = None
            if self.parents_available and batch.parents_available:
                self.parents = extend(self.parents, batch.parents)
            else:
                self.parents = None
            if self.parents_policy_available_1 and batch.parents_policy_available_1:
                self.parents_policy_1 = extend(self.parents_policy_1, batch.parents_policy_1)
            else:
                self.parents_policy_1 = None
            if self.parents_policy_available_2 and batch.parents_policy_available_2:
                self.parents_policy_2 = extend(self.parents_policy_2, batch.parents_policy_2)
            else:
                self.parents_policy_2 = None
            if self.parents_all_available and batch.parents_all_available:
                self.parents_all = extend(self.parents_all, batch.parents_all)
            else:
                self.parents_all = None
            if self.rewards_available and batch.rewards_available:
                self.rewards = extend(self.rewards, batch.rewards)
            else:
                self.rewards = None
        assert self.is_valid()
        return self

    def is_valid(self) -> bool:
        if len(self.states) != len(self):
            return False
        if len(self.actions_1) != len(self):
            return False
        if len(self.actions_2) != len(self):
            return False
        if len(self.done) != len(self):
            return False
        if len(self.traj_indices) != len(self):
            return False
        if len(self.state_indices) != len(self):
            return False
        if set(np.unique(self.traj_indices)) != set(self.envs.keys()):
            return False
        if set(self.trajectories.keys()) != set(self.envs.keys()):
            return False
        batch_indices = [
            idx for indices in self.trajectories.values() for idx in indices
        ]
        if len(batch_indices) != len(self):
            return False
        if len(np.unique(batch_indices)) != len(batch_indices):
            return False
        return True

    def traj_indices_are_consecutive(self) -> bool:
        trajectories_consecutive = list(self.trajectories) == list(
            np.arange(self.get_n_trajectories())
        )
        envs_consecutive = list(self.envs) == list(np.arange(self.get_n_trajectories()))
        return trajectories_consecutive and envs_consecutive

    def make_indices_consecutive(self):
        if self.traj_indices_are_consecutive():
            return
        self.traj_indices = self.get_trajectory_indices(consecutive=True).tolist()
        self.trajectories = OrderedDict(
            zip(range(self.get_n_trajectories()), self.trajectories.values())
        )
        self.envs = OrderedDict(
            {idx: env.set_id(idx) for idx, env in enumerate(self.envs.values())}
        )
        assert self.traj_indices_are_consecutive()
        assert self.is_valid()

    def _shift_indices(self, traj_shift: int, batch_shift: int):
        if not self.is_valid():
            raise Exception("Batch is not valid before attempting indices shift")
        self.traj_indices = [idx + traj_shift for idx in self.traj_indices]
        self.trajectories = {
            traj_idx + traj_shift: list(map(lambda x: x + batch_shift, batch_indices))
            for traj_idx, batch_indices in self.trajectories.items()
        }
        self.envs = {
            k + traj_shift: env.set_id(k + traj_shift) for k, env in self.envs.items()
        }
        if not self.is_valid():
            raise Exception("Batch is not valid after performing indices shift")
        return self

    def get_item(
        self,
        item: str,
        env: GFlowNetEnv = None,
        traj_idx: int = None,
        action_1: Tuple[int]= None,
        action_idx_1: int = None,
        action_idx_2: int = None,
        backward: bool = False,
    ):
        # Preliminary checks
        if env is not None:
            if traj_idx is not None:
                assert (
                    env.id == traj_idx
                ), "env.id {env.id} different to traj_idx {traj_idx}."
            else:
                traj_idx = env.id
            if action_idx_1 is not None:
                assert (
                    env.n_actions == action_idx_1
                ), "env.n_actions {env.n_actions} different to action_idx {action_idx}."
            else:
                action_idx_1 = env.n_actions
            if action_idx_2 is not None:
                assert (
                    env.n_actions == action_idx_2
                ), "env.n_actions {env.n_actions} different to action_idx {action_idx}."
            else:
                action_idx_2 = env.n_actions
        else:
            assert (
                traj_idx is not None and action_idx_1 is not None and action_idx_2 is not None
            ), "Either env or traj_idx AND action_idx must be provided"
        if action_idx_1 == 0:
            if backward is False:
                if item == "state":
                    return self.source["state"]
                elif item == "mask_f_1" or item == "mask_forward_1":
                    return self.source["mask_forward_1"]
                elif item == "mask_f_2" or item == "mask_forward_2":
                    return self.source["mask_forward_2"]
                else:
                    raise ValueError(
                        "Only state or mask_forward are available for a fresh env "
                        "(action_idx = 0)"
                    )
        
        batch_idx = self.traj_idx_action_idx_to_batch_idx(
            traj_idx, action_idx_1, backward
        )
        if batch_idx is None:
            if env is None:
                raise ValueError(
                    "{item} not available for action {action_idx} of trajectory "
                    "{traj_idx} and no env was provided."
                )
            else:
                if item == "state":
                    return env.state
                elif item == "done":
                    return env.done
                elif item == "mask_f_1" or item == "mask_forward_1":
                    return env.get_mask_invalid_actions_forward_1()
                elif item == "mask_f_2" or item == "mask_forward_2":
                    return env.get_mask_invalid_actions_forward_2(action_1=action_1)
                elif item == "mask_b_1" or item == "mask_backward_1":
                    return env.get_mask_invalid_actions_backward_1()
                elif item == "mask_b_2" or item == "mask_backward_2":
                    return env.get_mask_invalid_actions_backward_2()
                else:
                    raise ValueError(
                        "Not available in the batch. item must be one of: state, done, "
                        "mask_f[orward] or mask_b[ackward]."
                    )
        if item == "state":
            return self.states[batch_idx]
        elif item == "parent":
            return self.parents[batch_idx]
        elif item == "action_1":
            return self.actions_1[batch_idx]
        elif item == "action_2":
            return self.actions_2[batch_idx]
        elif item == "done":
            return self.done[batch_idx]
        elif item == "mask_f_1" or item == "mask_forward_1":
            if self.masks_invalid_actions_forward_1[batch_idx] is None:
                state = self.states[batch_idx]
                done = self.done[batch_idx]
                self.masks_invalid_actions_forward_1[batch_idx] = self.envs[
                    traj_idx
                ].get_mask_invalid_actions_forward_1(state, done)
            return self.masks_invalid_actions_forward_1[batch_idx]
        elif item == "mask_f_2" or item == "mask_forward_2":
            if self.masks_invalid_actions_forward_2[batch_idx] is None:
                state = self.states[batch_idx]
                done = self.done[batch_idx]
                self.masks_invalid_actions_forward_2[batch_idx] = self.envs[
                    traj_idx
                ].get_mask_invalid_actions_forward_2(state, done, action_1)
            return self.masks_invalid_actions_forward_2[batch_idx]
        elif item == "mask_b_1" or item == "mask_backward_1":
            if self.masks_invalid_actions_backward_1[batch_idx] is None:
                state = self.states[batch_idx]
                done = self.done[batch_idx]
                self.masks_invalid_actions_backward_1[batch_idx] = self.envs[
                    traj_idx
                ].get_mask_invalid_actions_backward_1(state, done)
            return self.masks_invalid_actions_backward_1[batch_idx]
        elif item == "mask_b_2" or item == "mask_backward_2":
            if self.masks_invalid_actions_backward_2[batch_idx] is None:
                state = self.states[batch_idx]
                done = self.done[batch_idx]
                self.masks_invalid_actions_backward_2[batch_idx] = self.envs[
                    traj_idx
                ].get_mask_invalid_actions_backward_2(state, done)
            return self.masks_invalid_actions_backward_2[batch_idx]
        else:
            raise ValueError(
                "item must be one of: state, parent, action_1, action_2, done, mask_f_1[orward], mask_f_2[orward], mask_b_1[ackward] or "
                "mask_b_2[ackward]"
            )
