from gflownet.gflownet.proxy.base import Proxy
import torch
import numpy as np


class DropoutRegressor(Proxy):
    def __init__(self, regressor, num_dropout_samples, **kwargs) -> None:
        super().__init__(**kwargs)
        self.regressor = regressor
        self.num_dropout_samples = num_dropout_samples
        if hasattr(self.regressor, "load_model") and not self.regressor.load_model():
            raise FileNotFoundError
        elif hasattr(self.regressor, "load_model") == False:
            print("Model has not been loaded from path.")

    def __call__(self, inputs):
        self.regressor.eval()
        if isinstance(inputs, np.ndarray):
            inputs = torch.FloatTensor(inputs).to(self.device)
        self.regressor.model.train()
        with torch.no_grad():
            output = self.regressor.model(inputs).squeeze(-1)
        return output
