"""
Runnable script with hydra capabilities
"""
import os
import pickle
import random
import sys
import torch

import hydra
import pandas as pd

from gflownet.utils.common import chdir_random_subdir
from gflownet.utils.policy import parse_policy_config
from rdkit import RDLogger

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RDLogger.DisableLog('rdApp.*')

# +
@hydra.main(config_path="./config", config_name="sample", version_base="1.1")
def sample(config):
    #chdir_random_subdir()

    # Get current directory and set it as root log dir for Logger
    cwd = os.getcwd()
    config.logger.logdir.root = cwd
    print(f"\nLogging directory of this run:  {cwd}\n")

    # Reset seed for job-name generation in multirun jobs
    random.seed(None)
    # Set other random seeds
    set_seeds(config.seed)

    # Logger
    logger = hydra.utils.instantiate(config.logger, config, _recursive_=False)
    
    # The proxy is required in the env for scoring: might be an oracle or a model
    proxy = hydra.utils.instantiate(
        config.proxy,
        device=config.device,
        float_precision=config.float_precision,
    )
    
    # The proxy is passed to env and used for computing rewards
    env = hydra.utils.instantiate(
        config.env,
        proxy=proxy,
        device=config.device,
        float_precision=config.float_precision,
    )
    
    # The policy is used to model the probability of a forward/backward action
    forward_1_config = parse_policy_config(config.policy.policy_1, kind="forward")
    backward_1_config = parse_policy_config(config.policy.policy_1, kind="backward")

    forward_policy_1 = hydra.utils.instantiate(
        forward_1_config,
        env=env,
        device=config.device,
        float_precision=config.float_precision,
    )
    
    # Load the trained policy model weights
    model_weights = torch.load('path/to/SynGFN/logs/xx/ckpts/ck_f_1_0_iterxx.ckpt')
    if not model_weights:
        print("Model weights are empty. Please check the weight save path")
    forward_policy_1.model.load_state_dict(model_weights)
    
    backward_policy_1 = hydra.utils.instantiate(
        backward_1_config,
        env=env,
        device=config.device,
        float_precision=config.float_precision,
        base=forward_policy_1,
    )
    
    forward_2_config = parse_policy_config(config.policy.policy_2, kind="forward")
    backward_2_config = parse_policy_config(config.policy.policy_2, kind="backward")

    forward_policy_2 = hydra.utils.instantiate(
        forward_2_config,
        env=env,
        device=config.device,
        float_precision=config.float_precision,
    )
    # Load the trained policy model weights
    model_weights = torch.load('path/to/SynGFN/logs/xx/ckpts/ck_f_2_0_iterxx.ckpt')
    if not model_weights:
        print("Model weights are empty. Please check the weight save path")
    forward_policy_2.model.load_state_dict(model_weights)

    backward_policy_2 = hydra.utils.instantiate(
        backward_2_config,
        env=env,
        device=config.device,
        float_precision=config.float_precision,
        base=forward_policy_2,
    )

    gflownet = hydra.utils.instantiate(
        config.gflownet,
        device=config.device,
        float_precision=config.float_precision,
        env=env,
        forward_policy_1=forward_policy_1,
        backward_policy_1=backward_policy_1,
        forward_policy_2=forward_policy_2,
        backward_policy_2=backward_policy_2,
        buffer=config.env.buffer,
        logger=logger,
    )
    
    print("Starting sampling")
       
    # Sample from trained SynGFN
    if config.n_samples > 0 and config.n_samples <= 1e5:
        batch_sample = 64
        x_sampled = set()
        proxy_val = []
        trajs = []

        while len(x_sampled) < config.n_samples:
            n_forward = min(batch_sample, config.n_samples - len(x_sampled))

            batch, times = gflownet.sample_batch(n_forward=n_forward, train=False, sampling_method="policy")
            final_state = batch.get_terminating_states(proxy=True)
            proxy_values = env.oracle(final_state).tolist()  
            states = batch.get_terminating_states()  
            new_samples = [env.state2readable(x) for x in states]

            for i, sample in enumerate(new_samples):
                if sample not in x_sampled:
                    x_sampled.add(sample) 
                    trajs.append(states[i])
                    proxy_val.append(proxy_values[i])

        df = pd.DataFrame(
            {
                "trajectory": trajs,
                "readable": list(x_sampled),
                "proxy_vals": proxy_val,
            }
        )
        df.to_csv(logger.logdir / "samples.csv")  
        
    print("Complete sampling")
    # Close logger
    gflownet.logger.end()

def set_seeds(seed):
    import numpy as np
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    sample()
    sys.exit()