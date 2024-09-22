import torch
import pickle
from typing import List
from torchtyping import TensorType
import pandas as pd
from pathlib import Path


def define_metrics(logger):
    for k in [1, 10, 50, 100]:
        logger.define_metric(
            "mean_energy_top{}".format(k), step_metric="post_al_cum_cost"
        )
        logger.define_metric(
            "mean_pairwise_distance_top{}".format(k), step_metric="post_al_cum_cost"
        )
        logger.define_metric(
            "mean_min_distance_from_mode_topp{}".format(k),
            step_metric="post_al_cum_cost",
        )
        logger.define_metric(
            "mean_min_distance_from_D0_topp{}".format(k), step_metric="post_al_cum_cost"
        )


def get_diversity(samples, env, energies, modes):
    if hasattr(env, "get_pairwise_distance"):
        pairwise_dists = env.get_pairwise_distance(samples)
        pairwise_dists = torch.sort(pairwise_dists, descending=True)[0]
        if modes is not None:
            min_dist_from_mode = env.get_pairwise_distance(samples, modes)
            min_dist_from_mode = torch.sort(min_dist_from_mode, descending=False)[0]
        else:
            min_dist_from_mode = torch.zeros_like(energies)
    else:
        pairwise_dists = torch.zeros_like(energies)
        min_dist_from_mode = torch.zeros_like(energies)
    return pairwise_dists, min_dist_from_mode


def get_novelty(samples, env, logger):
    train_path = logger.data_path.parent / Path("data_train.csv")
    try:
        train_dataset = pd.read_csv(train_path, index_col=0)
    except:
        print("Train Dataset not found.")
        return None
    dataset_samples = train_dataset["samples"]
    dataset_states = [env.readable2state(sample) for sample in dataset_samples]
    dists_from_D0 = env.get_distance_from_D0(samples, dataset_states)
    dists_from_D0 = torch.sort(dists_from_D0, descending=True)[0]
    return dists_from_D0


def evaluate(
    samples,
    energies,
    maximize,
    cumulative_cost,
    logger,
    env,
    modes=None,
    extrema=None,
    proxy_extrema=None,
):
    define_metrics(logger)
    energies = torch.sort(energies, descending=maximize)[0]
    metrics_dict = {}
    for k in logger.oracle.k:
        print(f"\n Top-{k} Performance")
        idx_topk = torch.argsort(energies, descending=maximize)[:k].tolist()
        samples_topk = [samples[i] for i in idx_topk]
        mean_energy_topk = torch.mean(energies[:k])
        metrics_dict.update({"mean_energy_top{}".format(k): mean_energy_topk})
        if k != 1:
            pairwise_dist_topk, min_dist_from_mode_topk = get_diversity(
                samples_topk, env, energies[:k], modes
            )
            mean_pairwise_dist_topk = torch.mean(pairwise_dist_topk)
            mean_min_dist_from_mode_topk = torch.mean(min_dist_from_mode_topk)
            dists_from_D0 = get_novelty(samples_topk, env, logger)
            mean_dist_from_D0_topk = torch.mean(dists_from_D0)
            metrics_dict.update(
                {"min_distance_from_D0_top{}".format(k): mean_dist_from_D0_topk}
            )
        else:
            mean_pairwise_dist_topk = 0
            mean_min_dist_from_mode_topk = 0
            mean_dist_from_D0_topk = 0
        metrics_dict.update(
            {"mean_pairwise_distance_top{}".format(k): mean_pairwise_dist_topk}
        )
        metrics_dict.update(
            {
                "mean_min_distance_from_mode_top{}".format(
                    k
                ): mean_min_dist_from_mode_topk
            }
        )

        if logger.progress:
            print(f"\t Mean Energy: {mean_energy_topk}")
            print(f"\t Mean Pairwise Distance: {mean_pairwise_dist_topk}")
            print(f"\t Mean Min Distance from Mode: {mean_min_dist_from_mode_topk}")
            print(f"\t Mean Min Distance from D0: {mean_dist_from_D0_topk}")

    if extrema is not None and proxy_extrema is not None:
        simple_regret = abs(torch.mean(extrema - energies[0]))
        inference_regret = abs(
            torch.mean(proxy_extrema.to(energies.device) - energies[0])
        )
        metrics_dict.update({"simple_regret": simple_regret})
        metrics_dict.update({"inference_regret": inference_regret})
        if logger.progress:
            print(f"\t Simple Regret: {simple_regret}")
            print(f"\t Inference Regret: {inference_regret}")

    if logger.progress:
        print(f"\n Cumulative Cost: {cumulative_cost}")
    metrics_dict.update({"post_al_cum_cost": cumulative_cost})
    logger.log_metrics(metrics_dict, use_context=False)
