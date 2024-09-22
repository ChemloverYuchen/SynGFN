"""
Buffer class to handle train and test data sets, reply buffer, etc.
"""
import pickle

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import os

tmp_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(tmp_path))))
class Buffer:
    def __init__(
        self,
        env,
        make_train_test=False,
        replay_capacity=0,
        output_csv=None,
        data_path=None,
        train=None,
        test=None,
        logger=None,
        **kwargs,
    ):
        self.logger = logger
        self.env = env
        if train is not None and "type" in train:
            self.train_type = train.type
        else:
            self.train_type = None
        self.train, dict_tr = self.make_data_set(train)
        self.replay_capacity = replay_capacity
        self.main = pd.DataFrame(columns=["state", "traj", "reward", "proxy_val", "iter"])
        self.replay = pd.DataFrame(
            np.empty((self.replay_capacity, 5), dtype=object),
            columns=["state", "traj", "reward", "proxy_val", "iter"],
        )
        self.replay.reward = pd.to_numeric(self.replay.reward)
        self.replay.proxy_val = pd.to_numeric(self.replay.proxy_val)
        self.replay.reward = [-1 for _ in range(self.replay_capacity)]
        self.replay_states = {}
        self.replay_trajs_1 = {}
        self.replay_trajs_2 = {}
        self.replay_rewards = {}
        self.replay_readable = {}
        self.replay_pkl = self.logger.logdir / "replay.pkl"
        self.save_replay()
            
        if test is not None and "type" in test:
            self.test_type = test.type
        else:
            self.test_type = None
        self.test, dict_tt = self.make_data_set(test)
        if (
            self.test is not None
            and "output_csv" in test
            and test.output_csv is not None
        ):
            self.test.to_csv(test.output_csv)
        if dict_tt is not None and "output_pkl" in test and test.output_pkl is not None:
            with open(test.output_pkl, "wb") as f:
                pickle.dump(dict_tt, f)
                self.test_pkl = test.output_pkl
        else:
#             print(
#                 """
#             Important: test metrics will NOT be computed. In order to compute
#             test metrics the test configuration of the buffer should be complete and
#             feasible and an output pkl file should be defined in
#             env.buffer.test.output_pkl.
#             """
#             )
            self.test_pkl = None
        #Compute buffer statistics
        if self.train is not None:
            (
                self.mean_tr,
                self.std_tr,
                self.min_tr,
                self.max_tr,
                self.max_norm_tr,
            ) = self.compute_stats(self.train)
        if self.test is not None:
            self.mean_tt, self.std_tt, self.min_tt, self.max_tt, _ = self.compute_stats(
                self.test
            )

    def save_replay(self):
        with open(self.replay_pkl, "wb") as f:
            pickle.dump(
                {
                    "x": self.replay_states,
                    "readable": self.replay_readable,
                    "trajs_1": self.replay_trajs_1,
                    "trajs_2": self.replay_trajs_2,
                    "rewards": self.replay_rewards,
                },
                f,
            )

    def add(
        self,
        states,
        trajs_1,
        trajs_2,
        rewards,
        proxy_val,
        it,
        buffer="main",
        criterion="greater",
    ):
        if buffer == "main":
            self.main = pd.concat(
                [
                    self.main,
                    pd.DataFrame(
                        {
                            "state": [self.env.state2readable(s) for s in states],
                            "traj_1": [self.env.traj2readable(p) for p in trajs_1],
                            "traj_2": [self.env.traj2readable(p) for p in trajs_2],
                            "reward": rewards,
                            "proxy_val": proxy_val,
                            "iter": it,
                        }
                    ),
                ],
                axis=0,
                join="outer",
            )
        elif buffer == "replay" and self.replay_capacity > 0:
            if criterion == "greater":
                self.replay = self._add_greater(states, trajs_1, trajs_2, rewards, proxy_val, it)

    def _add_greater(
        self,
        states,
        trajs_1,
        trajs_2,
        rewards,
        proxy_val,
        it,
        allow_duplicate_states=False,
    ):
        for idx, (state, traj_1,traj_2, reward, proxy_val) in enumerate(
            zip(states, trajs_1, trajs_2, rewards, proxy_val)
        ):
            if not allow_duplicate_states:
                if isinstance(state, torch.Tensor):
                    is_duplicate = False
                    for replay_state in self.replay_states.values():
                        if torch.allclose(state, replay_state, equal_nan=True):
                            is_duplicate = True
                            break
                else:
                    is_duplicate = state in self.replay_states.values()
                if is_duplicate:
                    continue
            if (
                reward > self.replay.iloc[-1]["reward"]
                and traj_1 not in self.replay_trajs_1.values() and traj_2 not in self.replay_trajs_2.values()
            ):
                self.replay.iloc[-1] = {
                    "state": self.env.state2readable(state),
                    "traj": state,
                    "reward": reward,
                    "proxy_val": proxy_val,
                    "iter": it,
                }
                self.replay_states[(idx, it)] = state
                self.replay_trajs_1[(idx, it)] = traj_1
                self.replay_trajs_2[(idx, it)] = traj_2
                self.replay_rewards[(idx, it)] = reward
                self.replay_readable[(idx, it)] = self.env.state2readable(state)
                self.replay.sort_values(by="reward", ascending=False, inplace=True)
        self.save_replay()
        return self.replay

    def make_data_set(self, config):
        if config is None:
            return None, None
        elif "path" in config and config.path is not None:
            path = self.logger.logdir / Path("data") / config.path
            df = pd.read_csv(path, index_col=0)
            return df
        elif "type" not in config:
            return None, None
        elif config.type == "all" and hasattr(self.env, "get_all_terminating_states"):
            samples = self.env.get_all_terminating_states()
        elif (
            config.type == "grid"
            and "n" in config
            and hasattr(self.env, "get_grid_terminating_states")
        ):
            samples = self.env.get_grid_terminating_states(config.n)
        elif (
            config.type == "uniform"
            and "n" in config
            and "seed" in config
            and hasattr(self.env, "get_uniform_terminating_states")
        ):
            samples = self.env.get_uniform_terminating_states(config.n, config.seed)
        elif (
            config.type == "random"
            and "n" in config
            and hasattr(self.env, "get_random_terminating_states")
        ):
            samples = self.env.get_random_terminating_states(config.n)
        else:
            return None, None
        
        proxy_val = self.env.oracle(self.env.statebatch2oracle(samples)).tolist()
        df = pd.DataFrame(
            {
                "samples": [self.env.state2readable(s) for s in samples],
                "proxy_val": proxy_val,
            }
        )
        return df, {"x": samples, "proxy_val": proxy_val}

    def compute_stats(self, data):
        mean_data = data["proxy_val"].mean()
        std_data = data["proxy_val"].std()
        min_data = data["proxy_val"].min()
        max_data = data["proxy_val"].max()
        data_zscores = (data["proxy_val"] - mean_data) / std_data
        max_norm_data = data_zscores.max()
        return mean_data, std_data, min_data, max_data, max_norm_data
