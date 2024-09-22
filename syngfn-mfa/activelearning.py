import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import sys
import os
import random
from typing import List
import hydra
from omegaconf import OmegaConf
import torch
from env.mfenv import MultiFidelityEnvWrapper
import matplotlib.pyplot as plt
import numpy as np
from utils.common import get_figure_plots
from utils.eval_al_round import evaluate
import pickle
from gflownet.gflownet.utils.common import chdir_random_subdir
from gflownet.gflownet.utils.policy import parse_policy_config

root_path = os.path.dirname(os.path.abspath(__file__))
pretrain_path = os.path.join(root_path, 'gflownet/pretrain/pretrain_batch_150mw_1k.ckpt')

@hydra.main(config_path="./config", config_name="default")
def main(config):
    if config.logger.logdir.root != "./logs":
        os.chdir(config.logger.logdir.root)
    cwd = os.getcwd()
    config.logger.logdir.root = cwd

    print(f"\nLogging directory of this run:  {cwd}\n")
    # Reset seed for job-name generation in multirun jobs
    random.seed(None)
    # Set other random seeds
    set_seeds(config.seed)

    print(
        "\n \tUser-Defined Warning: Oracles must be in increasing order of fidelity. \n \t Best oracle should be the last one in the config list."
    )

    # Logger
    logger = hydra.utils.instantiate(config.logger, config, _recursive_=False)
    
    N_FID = len(config._oracle_dict)
        
    if "true_oracle" in config:
        true_oracle = hydra.utils.instantiate(
            config.true_oracle,
            device=config.device,
            float_precision=config.float_precision,
        )
    else:
        true_oracle = None

    env = hydra.utils.instantiate(
        config.env,
        oracle=true_oracle,
        device=config.device,
        float_precision=config.float_precision,
    )
    
    if hasattr(env, "rescale"):
        rescale = env.rescale
    else:
        rescale = 1.0
        
    oracles = []
    for fid in range(1, N_FID + 1):
        oracle = hydra.utils.instantiate(
            config._oracle_dict[str(fid)],
            oracle=true_oracle,
            env=env,
            device=config.device,
            float_precision=config.float_precision,
        )
        oracles.append(oracle)
        
    # Update env to mfenv
    if N_FID > 1:
        env = MultiFidelityEnvWrapper(
            env,
            n_fid=N_FID,
            oracle=oracles,
            proxy_state_format=config.env.proxy_state_format,
            rescale=rescale,
            device=config.device,
            float_precision=config.float_precision,
            fid_embed=config.multifidelity.fid_embed,# one_hot
            fid_embed_dim=config.multifidelity.fid_embed_dim,# None
        )
        # Best fidelity
        env.env.oracle = oracles[-1]
    else:
        oracle = oracles[0]
        env.oracle = oracle
    config_model = None
    modes = None
    extrema = None
    proxy_extrema = None
    maximize = None
    
    # The policy is used to model the probability of a forward/backward action
    forward_1_config = parse_policy_config(config.policy.policy_1, kind="forward")
    backward_1_config = parse_policy_config(config.policy.policy_1, kind="backward")

    forward_policy_1 = hydra.utils.instantiate(
        forward_1_config,
        env=env,
        device=config.device,
        float_precision=config.float_precision,
    )
    
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
    
    # [Recommend] load pretrained policy model 2
#     model_weights = torch.load(pretrain_path)
#     forward_policy_2.model.load_state_dict(model_weights)
    
    backward_policy_2 = hydra.utils.instantiate(
        backward_2_config,
        env=env,
        device=config.device,
        float_precision=config.float_precision,
        base=forward_policy_2,
    )
    
    if "model" in config:
        config_model = config.model

    if hasattr(oracle, "modes"):
        modes = oracle.modes

    if hasattr(oracle, "extrema"):
        extrema = oracle.extrema

    # Budget
    if "budget" in config:
        BUDGET = config.budget
    else:
        if N_FID == 1:
            BUDGET = config.al_n_rounds * oracle.cost * config.n_samples
        else:
            BUDGET = oracles[-1].cost * config.al_n_rounds * config.n_samples
        
    if "proxy" in config and ("mes" or "kg" in config.proxy._target_.lower()):
        is_mes = True
    else:
        is_mes = False

    # Dataset for regressor
    data_handler = hydra.utils.instantiate(
        config.dataset,
        env=env,
        logger=logger,
        oracle=oracle,
        device=config.device,
        float_precision=config.float_precision,
        rescale=rescale,
        is_mes=is_mes,
    )
    
    if logger.resume == False:
        cumulative_cost = 0.0
        cumulative_sampled_states = []
        cumulative_sampled_samples = []
        cumulative_sampled_energies = torch.tensor(
            [], device=env.device, dtype=env.float
        )
        cumulative_sampled_fidelities = torch.tensor(
            [], device=env.device, dtype=env.float
        )
        iter = 1
    else:
        cumulative_cost = logger.resume_dict["cumulative_cost"]
        cumulative_sampled_states = logger.resume_dict["cumulative_sampled_states"]
        cumulative_sampled_samples = logger.resume_dict["cumulative_sampled_samples"]
        cumulative_sampled_energies = logger.resume_dict["cumulative_sampled_energies"]
        cumulative_sampled_fidelities = logger.resume_dict[
            "cumulative_sampled_fidelities"
        ]
        iter = logger.resume_dict["iter"] + 1
    
    # Reward func set
    env.reward_norm = env.reward_norm * env.norm_factor
    initial_reward_beta = env.reward_beta
    initial_reward_norm = env.reward_norm
    while cumulative_cost < BUDGET:
        # BETA and NORM SCHEDULING
        env.reward_beta = initial_reward_beta + (
            initial_reward_beta * env.beta_factor * (iter - 1)
        )
        # if norm_factor != 1, reward_norm ⬇ as al run 
        env.reward_norm = initial_reward_norm / (env.norm_factor ** (iter - 1))
        if config.multifidelity.proxy == True:
            regressor = hydra.utils.instantiate(
                config.regressor,
                env = env,
                dataset=data_handler,
                device=config.device,
                maximize=oracle.maximize,
                float_precision=config.float_precision,
                logger=logger,
            )
        print(f"\nStarting iteration {iter} of active learning")
        if logger:
            logger.set_context(iter)
        if N_FID == 1 or config.multifidelity.proxy == True:
            regressor.fit()
            logger.set_proxy_path()
            logger.save_proxy(regressor.model)
            
        # Proxy
        if "proxy" in config:
            proxy = hydra.utils.instantiate(
                config.proxy,
                regressor=regressor,
                device=config.device,
                float_precision=config.float_precision,
                logger=logger,
                oracle=oracles,
                env=env,
            )
        else:
            print('config.proxy is None')
            proxy = None
            
        env.set_proxy(proxy)
        
        # SynGFN
        gflownet = hydra.utils.instantiate(
            config.gflownet,
            env=env,
            buffer=config.env.buffer,
            logger=logger,
            device=config.device,
            forward_policy_1=forward_policy_1,
            backward_policy_1=backward_policy_1,
            forward_policy_2=forward_policy_2,
            backward_policy_2=backward_policy_2,
            float_precision=config.float_precision,
        )
        gflownet.train()
        
        if config.n_samples > 0 and config.n_samples <= 1e5:
            print("Sampling trajectories for active learning")
            
            batch, times = gflownet.sample_batch(
                n_forward=config.n_samples * 5, train=False
            )
            states = batch.get_terminating_states()
            
            if isinstance(states[0], list):
                states_tensor = torch.tensor(states)
            else:
                states_tensor = torch.vstack(states)
            states_tensor = states_tensor.unique(dim=0)
            if isinstance(states[0], list):
                # for envs in which we want list of lists
                states = states_tensor.tolist()
            else:
                # for the envs in which we want list of tensors
                states = list(states_tensor)
            state_proxy = env.statebatch2proxy(states)
            
            if isinstance(state_proxy, list):
                state_proxy = torch.FloatTensor(state_proxy).to(config.device)
                
            #Scores
            if proxy is not None:
                scores = env.proxy(state_proxy)
            else:
                scores, _ = regressor.get_predictions(env, states, denorm=True)
                
            # Pick top_k from 5*k samples
            num_pick = min(config.n_samples, len(states))
            maximize = True
            idx_pick = torch.argsort(scores, descending=maximize)[:num_pick].tolist()
            picked_states = [states[i] for i in idx_pick]
            if extrema is not None:
                proxy_extrema, _ = regressor.get_predictions(
                    env, picked_states[0], denorm=True
                )

            if N_FID > 1:
                picked_samples, picked_fidelity = env.statebatch2oracle(picked_states)
                picked_energies = env.call_oracle_per_fidelity(
                    picked_samples, picked_fidelity
                )

            else:
                picked_samples = env.statebatch2oracle(picked_states)
                picked_energies = env.oracle(picked_samples)
                picked_fidelity = None

            if hasattr(env, "get_cost"):
                cost_al_round = env.get_cost(picked_states, picked_fidelity)
                cumulative_cost += np.sum(cost_al_round)
                avg_cost = np.mean(cost_al_round)

            else:
                cost_al_round = torch.ones(len(picked_states))
                if hasattr(oracle, "cost"):
                    cost_al_round = cost_al_round * oracle.cost
                avg_cost = torch.mean(cost_al_round).detach().cpu().numpy()
                cumulative_cost += torch.sum(cost_al_round).detach().cpu().numpy()
              
            if N_FID == 1 or config.multifidelity.proxy == True:
                data_handler.update_dataset(
                    picked_states, picked_energies.tolist(), picked_fidelity
                )

            print("finished sampling")
        del gflownet
        del proxy
        del regressor
        env._test_traj_list = []
        env._test_traj_actions_list = []
        iter += 1


def set_seeds(seed):
    import torch
    import numpy as np

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
    sys.exit()
