from gflownet.gflownet.utils.logger import Logger
import torch
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb


class AL_Logger(Logger):
    def __init__(
        self,
        config,
        ckpts: dict,
        logdir: dict,
        **kwargs,
    ):
        super().__init__(config, logdir=logdir, checkpoints=ckpts.policy, **kwargs)
        self.proxy_period = (
            np.inf
            if ckpts.regressor.period == None or ckpts.regressor.period == -1
            else ckpts.regressor.period
        )
        self.data_dir = self.logdir / logdir.data
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def set_proxy_path(self, ckpt_id: str = None):
        if ckpt_id is None:
            self.proxy_ckpt_path = self.ckpts_dir / "ckpts"
        else:
            self.proxy_ckpt_path = self.ckpts_dir / f"{ckpt_id}"
    def save_proxy(self, model):
        if not int(self.context) % self.proxy_period:
            if self.proxy_ckpt_path is not None:
                ckpt_id = "regressor_iter{:03d}".format(int(self.context))
                stem = Path(
                    ckpt_id + ".ckpt"
                )
                path = self.proxy_ckpt_path.parent / stem
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                    },
                    path,
                )

    def log_dataset_stats(self, train_stats, test_stats):
        if not self.do.online:
            return
        for key in train_stats.keys():
            self.log_metric("train_" + key, train_stats[key], use_context=False)
            if test_stats is not None:
                self.log_metric("test_" + key, test_stats[key], use_context=False)

    def set_data_path(self, data_path: str = None):
        if data_path is None:
            self.data_path = None
        else:
            self.data_path = self.data_dir / f"{data_path}"

    def save_dataset(self, dataset, type):
        if self.data_path is not None:
            data = pd.DataFrame(dataset)
            data = data.sort_values(by="energies", ascending=False)
            if type == "sampled":
                type = type + "_iter" + self.context
            name = Path(self.data_path.stem + "_" + type + ".csv")
            path = self.data_path.parent / name
            data.to_csv(path)
        if self.do.online:
            wandb_path = Path(
                self.wandb.run.dir + "/" + self.data_path.stem + "_" + type + ".csv"
            )
            data.to_csv(wandb_path)

    def log_figure(self, key, fig, use_context):
        if not self.do.online and fig is not None:
            plt.close()
            return
        if use_context:
            key = self.context + "/" + key
        if fig is not None:
            figimg = self.wandb.Image(fig)
            self.wandb.log({key: figimg})
            plt.close()

    def define_metric(self, metric, step_metric=None):
        if not self.do.online:
            return
        wandb.define_metric(metric, step_metric=step_metric)
