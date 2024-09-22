import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.optim import Adam
import hydra
from tqdm import tqdm
from gflownet.gflownet.utils.common import set_device, set_float_precision


class DropoutRegressor:
    def __init__(
        self,
        device,
        float_precision,
        checkpoint,
        eps,
        max_epochs,
        history,
        lr,
        weight_decay,
        beta1,
        beta2,
        dataset,
        config_model,
        config_env,
        logger=None,
        **kwargs
    ):
        self.logger = logger
        self.config_model = config_model
        self.config_env = config_env

        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_precision)

        # Training Parameters
        self.eps = eps
        self.max_epochs = max_epochs
        self.history = history
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2

        assert self.history <= self.max_epochs

        # Dataset
        self.dataset = dataset
        self.n_fid = self.dataset.n_fid

        # Logger
        self.progress = self.logger.progress
        if checkpoint:
            self.logger.set_proxy_path(checkpoint)

    def initialize_model(self):
        """
        Initialize the network (MLP, Transformer, RNN)
        """
        self.model = (
            hydra.utils.instantiate(
                self.config_model,
                n_fid=self.n_fid,
                config_env=self.config_env,
                _recursive_=False,
                device=self.device,
            )
            .to(self.device)
            .to(self.float)
        )
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(self.beta1, self.beta2),
        )

    def load_model(self):
        """
        Load and returns the model
        """
        name = (
            self.logger.proxy_ckpt_path.stem + self.logger.context + "final" + ".ckpt"
        )
        path = self.logger.proxy_ckpt_path.parent / name

        self.initialize_model()
        if os.path.exists(path):
            # make the following line cpu compatible
            checkpoint = torch.load(path, map_location="cuda:0")
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.model.to(self.device).to(self.float)
            for state in self.optimizer.state.values():  # move optimizer to GPU
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            return True
        else:
            raise FileNotFoundError

    def fit(self):
        """
        Initialises the model and dataloaders.
        Trains the model and saves it once convergence is attained.
        """
        # we reset the model, cf primacy bias, here we train on more and more data
        self.initialize_model()

        # for statistics we save the tr and te errors
        [self.err_train_hist, self.err_test_hist] = [[], []]

        # get training data in torch format
        train_loader, test_loader = self.dataset.get_dataloader()

        pbar = tqdm(range(1, self.max_epochs + 1), disable=not self.progress)
        self.converged = 0

        for epoch in pbar:
            self.test(test_loader)

            self.logger.save_proxy(self.model, self.optimizer, final=False, epoch=epoch)

            self.train(train_loader)

            # after training at least "history" epochs, check convergence
            if epoch > self.history:
                self.check_convergence(epoch)
                if self.converged == 1:
                    self.logger.save_proxy(
                        self.model, self.optimizer, final=True, epoch=epoch
                    )
                    if self.progress:
                        print(
                            "Convergence reached in {} epochs with MSE {:.4f}".format(
                                epoch, self.err_test_hist[-1]
                            )
                        )
                    break

            if self.progress:
                description = "Train MSE: {:.4f} | Test MSE: {:.4f}".format(
                    self.err_train_hist[-1], self.err_test_hist[-1]
                )
                pbar.set_description(description)

    def train(self, train_loader):
        """
        Args:
            train-loader
        """
        err_train = []
        self.model.train(True)
        for x_batch, y_batch in tqdm(train_loader, disable=True):
            # Move self.device to class dataset instead
            output = self.model(x_batch.to(self.device))
            loss = F.mse_loss(output[:, 0], y_batch.to(self.device))
            if self.logger:
                self.logger.log_metric("proxy_train_mse", loss.item())
            err_train.append(loss.data)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.err_train_hist.append(
            torch.mean(torch.stack(err_train)).cpu().detach().numpy()
        )

    def test(self, test_loader):
        err_test = []
        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch in tqdm(test_loader, disable=True):
                output = self.model(x_batch.to(self.device))
                loss = F.mse_loss(output[:, 0], y_batch.to(self.device))
                if self.logger:
                    self.logger.log_metric("proxy_val_mse", loss.item())
                err_test.append(loss.data)
        self.err_test_hist.append(
            torch.mean(torch.stack(err_test)).cpu().detach().numpy()
        )

    def check_convergence(self, epoch):
        eps = self.eps
        history = self.history
        max_epochs = self.max_epochs

        if all(
            np.asarray(self.err_test_hist[-history + 1 :])
            > self.err_test_hist[-history]
        ):  # early stopping
            self.converged = 1  # not a legitimate criteria to stop convergence ...
            print("\nTest loss increasing.")

        if (
            abs(
                self.err_test_hist[-history] - np.average(self.err_test_hist[-history:])
            )
            / self.err_test_hist[-history]
            < eps
        ):
            self.converged = 1
            if self.progress:
                print("\nHit test loss convergence criterion.")

        if epoch >= max_epochs:
            self.converged = 1
            if self.progress:
                print("\nReached max_epochs.")

    def forward_with_uncertainty(self, x, num_dropout_samples=10):
        self.model.train()
        with torch.no_grad():
            outputs = torch.hstack([self.model(x) for _ in range(num_dropout_samples)])
        return outputs
