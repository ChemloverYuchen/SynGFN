import copy
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.special import logsumexp
from torch.distributions import Bernoulli
from tqdm import tqdm
from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem

from gflownet.envs.building_block import BuildingBlock
from gflownet.policy.building_block import Policy
from gflownet.utils.logger import Logger
from gflownet.utils.batch import Batch
from gflownet.utils.buffer import Buffer
from gflownet.utils.common import (
    batch_with_rest,
    set_device,
    set_float_precision,
    tbool,
    tfloat,
    tlong,
    torch2np,
)

class GFlowNetAgent:
    def __init__(
        self,
        env: BuildingBlock,
        seed,
        device,
        float_precision,
        optimizer,
        buffer: Buffer,
        forward_policy_1: Policy,
        forward_policy_2: Policy,
        backward_policy_1: Policy,
        backward_policy_2: Policy,
        mask_invalid_actions,
        temperature_logits,
        random_action_prob,
        pct_offline,
        logger: Logger,
        num_empirical_loss,
        oracle,
        active_learning=False,
        sample_only=False,
        replay_sampling="permutation",
        **kwargs,
    ):
        # Seed
        self.rng = np.random.default_rng(seed)
        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_precision)
        # Environment
        self.env = env
        # Loss
        if optimizer.loss in ["trajectorybalance", "tb"]:
            self.loss = "trajectorybalance"
            self.logZ = nn.Parameter(torch.ones(optimizer.z_dim) * 150.0 / 75)
        else:
            print("Unkown loss. Using flowmatch as default")
            self.loss = "flowmatch"
            self.logZ = None
          
        # Logging
        self.num_empirical_loss = num_empirical_loss
        self.logger = logger
        self.oracle_n = oracle.n #number of samples for oracle metrics
        
        # Buffers
        self.replay_sampling = replay_sampling
        self.buffer = Buffer(
            **buffer, env=self.env, make_train_test=not sample_only, logger=logger
        )
        
        energies_stats_tr = None
        if self.env.reward_norm_std_mult > 0 and energies_stats_tr is not None:
            self.env.reward_norm = self.env.reward_norm_std_mult * energies_stats_tr[3]
            self.env.set_reward_norm(self.env.reward_norm)
       
        # Policy models_1
        self.forward_policy_1 = forward_policy_1
        if self.forward_policy_1.checkpoint is not None:
            self.logger.set_forward_policy_ckpt_path_1(self.forward_policy_1.checkpoint)
            if False:
                self.forward_policy_1.load_state_dict(
                    torch.load(self.policy_forward_path)
                )
                print("Reloaded GFN forward policy model Checkpoint")
        else:
            self.logger.set_forward_policy_ckpt_path_1(None)
        self.backward_policy_1 = backward_policy_1
        self.logger.set_backward_policy_ckpt_path_1(None)
        if self.backward_policy_1.checkpoint is not None:
            self.logger.set_backward_policy_ckpt_path_1(self.backward_policy_1.checkpoint)
            if False:
                self.backward_policy_1.load_state_dict(
                    torch.load(self.policy_backward_path)
                )
                print("Reloaded GFN backward policy model Checkpoint")
        else:
            self.logger.set_backward_policy_ckpt_path_1(None)
        
        # Policy models_2
        self.forward_policy_2 = forward_policy_2
        if self.forward_policy_2.checkpoint is not None:
            self.logger.set_forward_policy_ckpt_path_2(self.forward_policy_2.checkpoint)
            if False:
                self.forward_policy.load_state_dict(
                    torch.load(self.policy_forward_path)
                )
                print("Reloaded GFN forward policy model Checkpoint")
        else:
            self.logger.set_forward_policy_ckpt_path_2(None)
        self.backward_policy_2 = backward_policy_2
        self.logger.set_backward_policy_ckpt_path_2(None)
        if self.backward_policy_2.checkpoint is not None:
            self.logger.set_backward_policy_ckpt_path_2(self.backward_policy_2.checkpoint)
            if False:
                self.backward_policy.load_state_dict(
                    torch.load(self.policy_backward_path)
                )
                print("Reloaded GFN backward policy model Checkpoint")
        else:
            self.logger.set_backward_policy_ckpt_path_2(None)
            
        # Optimizer
        if self.forward_policy_1.is_model and self.forward_policy_2.is_model:         
            self.opt, self.lr_scheduler = make_opt(
                self.parameters(), self.logZ, optimizer
            )
        else:
            self.opt, self.lr_scheduler, self.target_1, self.target_2 = None, None, None, None
        self.n_train_steps = optimizer.n_train_steps
        self.batch_size = optimizer.batch_size
        self.batch_size_total = sum(self.batch_size.values())
        self.ttsr = max(int(optimizer.train_to_sample_ratio), 1)
        self.sttr = max(int(1 / optimizer.train_to_sample_ratio), 1)
        self.clip_grad_norm = optimizer.clip_grad_norm
        self.tau = optimizer.bootstrap_tau
        self.ema_alpha = optimizer.ema_alpha
        self.early_stopping = optimizer.early_stopping
        self.use_context = active_learning
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        
        # Training
        self.mask_invalid_actions = mask_invalid_actions
        self.temperature_logits = temperature_logits
        self.random_action_prob = random_action_prob
        self.pct_offline = pct_offline
        
        # Metrics
        self.l1 = -1.0
        self.kl = -1.0
        self.jsd = -1.0
        self.corr_prob_traj_rewards = 0.0
        self.var_logrewards_logp = -1.0
        self.nll_tt = 0.0

    def parameters(self):
        if not self.backward_policy_1.is_model and not self.backward_policy_2.is_model:
            return list(self.forward_policy_1.model.parameters())+list(self.forward_policy_2.model.parameters())
        elif self.loss == "trajectorybalance":
            return [param for policy in [self.forward_policy_1, self.backward_policy_1, self.forward_policy_2, self.backward_policy_2] for param in policy.model.parameters()]

        else:
            raise ValueError("Backward Policy cannot be a nn in flowmatch.")

    def sample_actions_1(
        self,
        envs: List[BuildingBlock],
        batch: Optional[Batch] = None,
        sampling_method: Optional[str] = "policy",
        backward: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        random_action_prob: Optional[float] = None,
        no_random: Optional[bool] = True,
        times: Optional[dict] = None,
    ) -> List[Tuple]:       
        # Preliminaries
        if sampling_method == "random":
            assert (
                no_random is False
            ), "sampling_method random and no_random True is ambiguous"
            random_action_prob = 1.0
            temperature = 1.0
        elif no_random is True:
            temperature = 1.0
            random_action_prob = 0.0
        else:
            if temperature is None:
                temperature = self.temperature_logits
            if random_action_prob is None:
                random_action_prob = self.random_action_prob
        if backward:
            model = self.backward_policy_1
        else:
            model = self.forward_policy_1

        if not isinstance(envs, list):
            envs = [envs]
        # Build states and masks
        states = [env.state for env in envs]
        
        if self.mask_invalid_actions is True:
            if batch is not None:
                if backward:#backward指replay
                    mask_invalid_actions = tbool(
                        [
                            batch.get_item("mask_backward_1", env, backward=True)
                            for env in envs
                        ],
                        device=self.device,
                    )
                else:
                    mask_invalid_actions = tbool(
                        [batch.get_item("mask_forward_1", env) for env in envs],
                        device=self.device,
                    )
            # Compute masks since a batch was not provided
            else:
                if backward:
                    mask_invalid_actions = tbool(
                        [env.get_mask_invalid_actions_backward_1() for env in envs],
                        device=self.device,
                    )
                else:
                    mask_invalid_actions = tbool(
                        [env.get_mask_invalid_actions_forward_1() for env in envs],
                        device=self.device,
                    )
        else:
            mask_invalid_actions = None

        # Build policy outputs
        policy_outputs = model.random_distribution_1(states)
        idx_norandom = (
            Bernoulli(
                (1 - random_action_prob) * torch.ones(len(states), device=self.device)
            )
            .sample()
            .to(bool)
        )
        # Get policy outputs from model
        if sampling_method == "policy":
            # Check for at least one non-random action
            if idx_norandom.sum() >0:
                states_policy = tfloat(
                    self.env.statebatch2policy_1(
                        [s for s, do in zip(states, idx_norandom) if do]
                    ),
                    device=self.device,
                    float_type=self.float,
                )
                policy_outputs[idx_norandom, :] = model(states_policy)
        else:
            raise NotImplementedError

        # Sample actions from policy outputs
        actions, logprobs = self.env.sample_actions_batch_1(
            policy_outputs=policy_outputs,
            mask=mask_invalid_actions,
            states_from=states,
            is_backward=backward,
            sampling_method=sampling_method,
            temperature_logits=temperature,
        )
        return actions
    
    def sample_actions_2(
        self,
        envs: List[BuildingBlock],
        batch: Optional[Batch] = None,
        actions: List[Tuple[int]] = None,
        #sampling_method: Optional[str] = "random",
        sampling_method: Optional[str] = "policy",
        backward: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        random_action_prob: Optional[float] = None,
        no_random: Optional[bool] = True,
        times: Optional[dict] = None,
    ) -> List[Tuple]:
        # Preliminaries
        if sampling_method == "random":
            assert (
                no_random is False
            ), "sampling_method random and no_random True is ambiguous"
            random_action_prob = 1.0
            temperature = 1.0
        elif no_random is True:
            temperature = 1.0
            random_action_prob = 0.0
        else:
            if temperature is None:
                temperature = self.temperature_logits
            if random_action_prob is None:
                random_action_prob = self.random_action_prob
        if backward:
            model = self.backward_policy_2
        else:
            model = self.forward_policy_2

        if not isinstance(envs, list):
            envs = [envs]
        # Build states and masks
        states = [env.state for env in envs]
 
        if self.mask_invalid_actions is True:
            if batch is not None:
                if backward:
                    mask_invalid_actions = tbool(
                        [
                            batch.get_item("mask_backward_2", env, backward=True)
                            for env in envs
                        ],
                        device=self.device,
                    )
                else:
                    mask_invalid_actions = tbool(
                        [batch.get_item("mask_forward_2", env, action_1=action) for env, action in zip(envs,actions)],
                        device=self.device,
                    )
            # Compute masks since a batch was not provided
            else:
                if backward:
                    mask_invalid_actions = tbool(
                        [env.get_mask_invalid_actions_backward_2() for env in envs],
                        device=self.device,
                    )
                else:
                    mask_invalid_actions = tbool(
                        [env.get_mask_invalid_actions_forward_2(action_1 = action) for env, action in zip(envs, actions)],
                        device=self.device,
                    )
        else:
            mask_invalid_actions = None

        # Build policy outputs
        policy_outputs = model.random_distribution_2(states)
        idx_norandom = (
            Bernoulli(
                (1 - random_action_prob) * torch.ones(len(states), device=self.device)
            )
            .sample()
            .to(bool)
        )
        # Get policy outputs from model
        if sampling_method == "policy":
            # Check for at least one non-random action
            if idx_norandom.sum() > 0:
                states_policy = tfloat(
                    self.env.statebatch2policy_2(
                        [s for s, do in zip(states, idx_norandom) if do], [a for a, do in zip(actions, idx_norandom) if do]
                    ),
                    device=self.device,
                    float_type=self.float,
                )
                policy_outputs[idx_norandom, :] = model(states_policy)
        else:
            raise NotImplementedError

        # Sample actions from policy outputs
        actions, _  = self.env.sample_actions_batch_2(
            policy_outputs=policy_outputs,
            mask=mask_invalid_actions,
            states_from=states,
            is_backward=backward,
            sampling_method=sampling_method,
            temperature_logits=temperature,
            #sample_only = no_random
            )
        return actions
    
    def step(
        self,
        envs: List[BuildingBlock],
        actions_1: List[Tuple],
        actions_2: List[Tuple],
        backward: bool = False,
    ):
        try:
            assert len(envs) == len(actions_1)
            assert len(envs) == len(actions_2)
        except:
            print(len(envs), len(actions_1), len(actions_2))
        assert len(envs) == len(actions_1)
        assert len(envs) == len(actions_2)
        if not isinstance(envs, list):
            envs = [envs]
        if backward:
            next_step = [env.step(action_1=action_1,action_2=action_2,backward=True) for env, action_1, action_2 in zip(envs, actions_1, actions_2)]
            _, actions_1, actions_2, valids = zip( *next_step )
        
        else:
            next_step = [env.step(action_1,action_2) for env, action_1, action_2 in zip(envs, actions_1, actions_2)]
            _, actions_1, actions_2, valids = zip( *next_step )
        return envs, actions_1, actions_2, valids

    @torch.no_grad()
    def sample_batch(
        self,
        n_forward: int = 0, #number of forward trajectories
        n_train: int = 0, #number of backward trajectories
        n_replay: int = 0, #number of backward trajectories from replay buffer
        train=True,
        progress=False,
        sampling_method: Optional[str] = "policy",
    ):
        # PRELIMINARIES: Prepare Batch and environments
        times = {
            "all": 0.0,
            "forward_actions": 0.0,
            "train_actions": 0.0,
            "replay_actions": 0.0,
            "actions_envs": 0.0,
        }
        t0_all = time.time()
        batch = Batch(env=self.env, device=self.device, float_type=self.float)

        # ON-POLICY FORWARD trajectories
        t0_forward = time.time()
        envs = [self.env.copy().reset(idx) for idx in range(n_forward)]
        batch_forward = Batch(env=self.env, device=self.device, float_type=self.float)
        while len(envs) > 0:
            # Sample actions
            t0_a_envs = time.time()
            actions_1 = self.sample_actions_1(
                envs=envs,
                batch=batch_forward,
                no_random=not train,
                times=times,
                sampling_method=sampling_method,
            )
            actions_2 = self.sample_actions_2(
                envs=envs,
                batch=batch_forward,
                actions=actions_1,
                no_random=not train,
                times=times,
                sampling_method=sampling_method,
            )
            times["actions_envs"] += time.time() - t0_a_envs
            # Update environments with sampled actions
            envs, actions_1, actions_2, valids = self.step(envs, actions_1, actions_2)
            # Add to batch
            # If train is False, only the variables of terminating states are stored.
            batch_forward.add_to_batch(envs, actions_1, actions_2, valids, train=train)
            # Filter out finished trajectories
            envs = [env for env in envs if not env.done]
            
        times["forward_actions"] = time.time() - t0_forward

        # TRAIN BACKWARD trajectories(offline trajectories)
        t0_train = time.time()
        envs = [self.env.copy().reset(idx) for idx in range(n_train)]
        batch_train = Batch(env=self.env, device=self.device, float_type=self.float)
        if n_train > 0 and self.buffer.train_pkl is not None:
            with open(self.buffer.train_pkl, "rb") as f:
                dict_tr = pickle.load(f)
                x_tr = self.rng.permutation(dict_tr["x"])
        actions_1 = []
        actions_2 = []
        valids = [True]*len(envs)
        for idx, env in enumerate(envs):
            env.set_state(x_tr[idx].tolist(), done=True)
            actions_1.append((env.eos_1,))
            actions_2.append((env.eos_2,))
        if envs:
            batch_train.add_to_batch(envs, actions_1, actions_2, valids, backward=True, train=train)
            envs, actions_1, actions_2, valids = self.step(envs, actions_1, actions_2, backward=True)
            batch_train.add_to_batch(envs, actions_1, actions_2, valids, backward=True, train=train)
        while envs:
            # Sample backward actions
            t0_a_envs = time.time()
            actions_1 = self.sample_actions_1(
                envs,
                batch_train,
                backward=True,
                no_random=not train,
                times=times,
                sampling_method=sampling_method,
            )
            actions_2 = self.sample_actions_2(
                envs,
                batch_train,
                actions=actions_1,
                backward=True,
                no_random=not train,
                times=times,
                sampling_method=sampling_method,
            )
            times["actions_envs"] += time.time() - t0_a_envs
            # Update environments with sampled actions
            envs, actions_1,actions_2, valids = self.step(envs, actions_1, actions_2, backward=True)
            # Add to batch
            batch_train.add_to_batch(envs, actions_1, actions_2, valids, backward=True, train=train)
            #assert all(valids)
            # Filter out finished trajectories
            envs = [env for env in envs if not env.done]
        times["train_actions"] = time.time() - t0_train

        # REPLAY BACKWARD trajectories
        t0_replay = time.time()
        batch_replay = Batch(env=self.env, device=self.device, float_type=self.float)
        if n_replay > 0 and self.buffer.replay_pkl is not None:
            with open(self.buffer.replay_pkl, "rb") as f:
                dict_replay = pickle.load(f)
            n_replay = min(n_replay, len(dict_replay["x"]))
            envs = [self.env.copy().reset(idx) for idx in range(n_replay)]
            if n_replay > 0:
                x_replay = list(dict_replay["x"].values())
                if self.replay_sampling == "permutation":
                    x_replay = [
                        x_replay[idx] for idx in self.rng.permutation(n_replay)
                    ]
                elif self.replay_sampling == "weighted":
                    x_rewards = np.fromiter(
                        dict_replay["rewards"].values(), dtype=float
                    )
                    x_indices = np.random.choice(
                        len(x_replay),
                        size=n_replay,
                        replace=False,
                        p=x_rewards / x_rewards.sum(),
                    )
                    new_indices = []
                    padding_idx=self.env.get_padding_idx()
                    for idx in x_indices:
                        while x_replay[idx][1] == padding_idx:
                            idx = np.random.choice(len(x_replay), size=1)[0]
                        new_indices.append(idx)
                    x_indices = np.array(new_indices)
                    x_replay = [x_replay[idx] for idx in x_indices]
                else:
                    raise ValueError(
                        f"Unrecognized replay_sampling = {self.replay_sampling}."
                    )
            actions_1 = []
            actions_2 = []
            valids = [True]*len(envs)
            for idx, env in enumerate(envs):
                env.set_state(x_replay[idx], done=True)
                actions_1.append((env.eos_1,))
                actions_2.append((env.eos_2,))
                
        if envs:
            batch_replay.add_to_batch(envs, actions_1, actions_2, valids, backward=True, train=train)

            envs, actions_1, actions_2, valids = self.step(envs, actions_1, actions_2, backward=True)
            batch_replay.add_to_batch(envs, actions_1, actions_2, valids, backward=True, train=train)
        while envs:
            # Sample backward actions
            t0_a_envs = time.time()
            
            actions_1 = self.sample_actions_1(
                envs,
                batch_replay,
                backward=True,
                no_random=not train,
                times=times,
                sampling_method=sampling_method,
            )
            actions_2 = self.sample_actions_2(
                envs,
                batch_replay,
                actions=actions_1,
                backward=True,
                no_random=not train,
                times=times,
                sampling_method=sampling_method,
            )
            times["actions_envs"] += time.time() - t0_a_envs
            
            # Update environments with sampled actions
            envs, actions_1, actions_2, valids = self.step(envs, actions_1, actions_2, backward=True)

            # Add to batch
            batch_replay.add_to_batch(envs, actions_1, actions_2, valids, backward=True, train=train)
    
            # Filter out finished trajectories
            envs = [env for env in envs if not env.done]
        times["replay_actions"] = time.time() - t0_replay

        # Merge forward and backward batches
        batch = batch.merge([batch_forward, batch_train, batch_replay])

        times["all"] = time.time() - t0_all

        return batch, times

    
    def compute_logprobs_trajectories(self, batch: Batch, backward: bool = False):
        assert batch.is_valid()

        states_1 = batch.get_states_1(policy=True)
        states_2 = batch.get_states_2(policy=True)
        actions_1 = batch.get_actions_1()
        actions_2 = batch.get_actions_2()
        parents_1 = batch.get_parents_1(policy=True)
        parents_2 = batch.get_parents_2(policy=True)
        traj_indices = batch.get_trajectory_indices(consecutive=True)
        if backward:
            # Backward trajectories
            masks_b_1,masks_b_2 = batch.get_masks_backward()
            policy_output_b_1 = self.backward_policy_1(states_1)#backward的logit计算由backward model得到
            policy_output_b_2 = self.backward_policy_2(states_2)
            logprobs_states_1 = self.env.get_logprobs_1(
                policy_output_b_1, False, actions_1, parents_1, masks_b_1,self.temperature_logits
            )
            logprobs_states_2 = self.env.get_logprobs_2(
                policy_output_b_2, False, actions_2, parents_2, masks_b_2, self.temperature_logits
            )
            logprobs_states = torch.add(logprobs_states_1, logprobs_states_2)
        else:
            # Forward trajectories
            masks_f_1, masks_f_2 = batch.get_masks_forward(of_parents=True)
            policy_output_f_1 = self.forward_policy_1(parents_1)
            policy_output_f_2 = self.forward_policy_2(parents_2)
            logprobs_states_1 = self.env.get_logprobs_1(
                policy_output_f_1, True, actions_1, states_1, masks_f_1,self.temperature_logits
            )
            
            parents = batch.get_parents_1()
            
            idx = [parents[i] == parents[0] for i in range(len(batch))]
            logprobs_states_1[idx] = 0
            
            logprobs_states_2 = self.env.get_logprobs_2(
                policy_output_f_2, True, actions_2, states_2, masks_f_2, self.temperature_logits
            )
            logprobs_states = torch.add(logprobs_states_1, logprobs_states_2)
            
        # Sum log probabilities of all transitions in each trajectory
        logprobs = torch.zeros(
            batch.get_n_trajectories(),
            dtype=self.float,
            device=self.device,
        ).index_add_(0, traj_indices, logprobs_states)
        return logprobs

    def trajectorybalance_loss(self, it, batch: Batch):
        # Get logprobs of forward and backward transitions
        logprobs_f = self.compute_logprobs_trajectories(batch, backward=False)
        logprobs_b = self.compute_logprobs_trajectories(batch, backward=True)
        # Get rewards from batch
        rewards = batch.get_terminating_rewards(sort_by="trajectory")
        loss = (
            (self.logZ.sum() + logprobs_f - logprobs_b - torch.log(rewards))
            .pow(2)
            .mean()
        )
        return loss, loss, loss

    def train(self):
        # Metrics
        all_losses = []
        all_visited = []
        loss_term_ema = None
        loss_flow_ema = None
        # Train loop
        pbar = tqdm(range(1, self.n_train_steps + 1), disable=not self.logger.progress, dynamic_ncols=True)
        for it in pbar:
            t0_iter = time.time()
            batch = Batch(env=self.env, device=self.device, float_type=self.float)
            n_forward = self.batch_size.forward
            n_train = self.batch_size.backward_dataset
            n_replay = self.batch_size.backward_replay
            for j in range(self.sttr):
                sub_batch, times = self.sample_batch(
                    n_forward=self.batch_size.forward,
                    n_train=self.batch_size.backward_dataset,
                    n_replay=self.batch_size.backward_replay,
                )
                batch.merge(sub_batch)
            for j in range(self.ttsr):
                if self.loss == "trajectorybalance":
                    losses = self.trajectorybalance_loss(
                        it * self.ttsr + j, batch
                    )
                else:
                    print("Unknown loss!")
                if not all([torch.isfinite(loss) for loss in losses]):
                    if self.logger.debug:
                        print("Loss is not finite - skipping iteration")
                    if len(all_losses) > 0:
                        all_losses.append([loss for loss in all_losses[-1]])
                
                else:
                    losses[0].backward()
                    if self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.parameters(), self.clip_grad_norm
                        )
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()
                    all_losses.append([i.item() for i in losses])
            #Buffer
            t0_buffer = time.time()
            
            states_term = batch.get_terminating_states(sort_by="trajectory")
            
            rewards = batch.get_terminating_rewards(sort_by="trajectory")
            actions_trajectories_1 = batch.get_actions_trajectories_1()
            actions_trajectories_2 = batch.get_actions_trajectories_2()
            proxy_vals = self.env.reward2proxy(rewards).tolist()
            rewards = rewards.tolist()
            padding_idx=self.env.get_padding_idx()
            n = sum(1 for i in states_term if i[1] == padding_idx)

   
            #buffer.replay
            self.buffer.add(
                states_term,
                actions_trajectories_1,
                actions_trajectories_2,
                rewards,
                proxy_vals,
                it,
                buffer="replay",
            )
            t1_buffer = time.time()
            times.update({"buffer": t1_buffer - t0_buffer})
            # Log
            if self.logger.lightweight:
                all_losses = all_losses[-100:]
                all_visited = states_term
            else:
                all_visited.extend(states_term)
                
            mean_similarity = -1
            if it % 10 == 0:
                final_smiles = batch.get_terminating_states(proxy=True)[0:n_forward]
                rewards_f = rewards[0:n_forward]
                combined_data = list(zip(final_smiles, rewards_f))
                sorted_data = sorted(combined_data, key=lambda x: x[1], reverse=True)
                top_10_smiles = [smiles for smiles, _ in sorted_data[:10]]
                mol_list = [Chem.MolFromSmiles(smiles) for smiles in top_10_smiles]
                fp_list = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in mol_list]
                similarity_list = []
                for i, fp_i in enumerate(fp_list):
                    for j, fp_j in enumerate(fp_list[i+1:], i+1):
                        similarity = DataStructs.TanimotoSimilarity(fp_i, fp_j)
                        similarity_list.append(similarity)
                mean_similarity = np.mean(similarity_list)

            self.logger.progressbar_update(
                pbar, all_losses, rewards[0:n_forward], rewards[n_forward:n_forward+n_train], rewards[n_train+n_forward:], mean_similarity, it, self.use_context
            )    
            
            # Save intermediate models
            t0_model = time.time()
            self.logger.save_models(self.forward_policy_1, self.forward_policy_2, self.backward_policy_1, self.backward_policy_2, step=it)
            t1_model = time.time()
            times.update({"save_intermediate_model": t1_model - t0_model})

            # Moving average of the loss for early stopping
            if loss_term_ema and loss_flow_ema:
                loss_term_ema = (
                    self.ema_alpha * losses[1].item()
                    + (1.0 - self.ema_alpha) * loss_term_ema
                )
                loss_flow_ema = (
                    self.ema_alpha * losses[2].item()
                    + (1.0 - self.ema_alpha) * loss_flow_ema
                )
                # Early stopping condition
                if (
                    loss_term_ema < self.early_stopping
                    and loss_flow_ema < self.early_stopping
                ):
                    break
            else:
                loss_term_ema = losses[1].item()
                loss_flow_ema = losses[2].item()

            # Log times
            t1_iter = time.time()
            iter_times = t1_iter - t0_iter
            times.update({"iter": iter_times})
            self.logger.log_time(times, use_context=self.use_context)

            
            self.record_steps(
                step=it,
                losses=all_losses,
                rewards_total=rewards,
                rewards_forward=rewards[0:n_forward],
                rewards_train=rewards[n_forward:n_train+n_forward],
                rewards_replay=rewards[n_train+n_forward:],
                mean_similarity=mean_similarity,
                save_file= self.logger.logdir / 'log_1.log',
            )
            
        # Save final model
        self.logger.save_models(self.forward_policy_1,self.forward_policy_2, self.backward_policy_1,self.backward_policy_2, final=True)
        # Close logger
        if self.use_context is False:
            self.logger.end()
            
    def record_steps(self, step: int, losses, rewards_total,rewards_forward,rewards_train,rewards_replay, mean_similarity, save_file: str, n_mean: int=100, top_k: int=10):
        loss = np.mean(np.array(losses)[-n_mean:, 0], axis=0)
        mean_reward_total = np.mean(np.array(rewards_total))
        mean_reward_forward = np.mean(np.array(rewards_forward))
        mean_reward_train = np.mean(np.array(rewards_train)) if rewards_train else -1
        mean_reward_replay = np.mean(np.array(rewards_replay)) if rewards_replay else -1
        top_k_r = np.mean(np.sort(rewards_forward)[len(rewards_forward) - top_k:])
        description = f'{step},{loss},{mean_reward_total},{mean_reward_forward},{mean_reward_train},{mean_reward_replay},{top_k_r},{mean_similarity}\n'
        if not os.path.exists(save_file):
            with open(save_file, 'w+') as f:
                f.write(f'iteration,loss,total mean reward,forward mean reward,train mean reward,replay mean reward,top_{top_k} mean forward reward,mean similarity\n')
                f.write(description)
        else:
            with open(save_file, 'a+') as f:
                f.write(description)

    def log_iter(
        self,
        pbar,
        rewards,
        proxy_vals,
        states_term,
        data,
        it,
        times,
        losses,
        all_losses,
        all_visited,
    ):
        # train metrics
        self.logger.log_sampler_train(
            rewards, proxy_vals, states_term, data, it, self.use_context
        )

        # logZ
        self.logger.log_metric("logZ", self.logZ.sum(), it, use_context=False)

        # test metrics
        if not self.logger.lightweight and self.buffer.test is not None:
            corr, data_logq, times = self.get_log_corr(times)
            self.logger.log_sampler_test(corr, data_logq, it, self.use_context)

        # oracle metrics
        oracle_batch, oracle_times = self.sample_batch(
            n_forward=self.oracle_n, train=False
        )

        if not self.logger.lightweight:
            self.logger.log_metric(
                "unique_states",
                np.unique(all_visited).shape[0],
                step=it,
                use_context=self.use_context,
            )


def make_opt(params, logZ, config):
    """
    Set up the optimizer
    """
    params = params
    if not len(params):
        return None
    if config.method == "adam":
        opt = torch.optim.Adam(
            params,
            config.lr,
            betas=(config.adam_beta1, config.adam_beta2),
        )
        if logZ is not None:
            opt.add_param_group(
                {
                    "params": logZ,
                    "lr": config.lr * config.lr_z_mult,
                }
            )
    elif config.method == "msgd":
        opt = torch.optim.SGD(params, config.lr, momentum=config.momentum)
    # Learning rate scheduling
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        opt,
        step_size=config.lr_decay_period,
        gamma=config.lr_decay_gamma,
    )
    return opt, lr_scheduler