import datetime
import logging

import os
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from typing import List

import omegaconf


from active_fair_ranking.algorithms.baseline import baseline
from active_fair_ranking.algorithms.data import SetofItems, create_ranking
from active_fair_ranking.algorithms.utils import (
    _set_seed,
    create_random_prot_attrs,
    setup_logging,
)

from active_fair_ranking.algorithms.pairwise import pairwise
from active_fair_ranking.algorithms.ranking_algorithms import ARalg


def read_config():
    config_file = "config_baseline.yaml"
    config = omegaconf.OmegaConf.load(config_file)

    # setup the logger
    logger = setup_logging(config)

    # write post init checks here

    assert (
        config.beat_the_pivot.subset_size <= config.num_of_items
    ), f"subset size {config.subset_size} must be <= number of items {config.num_of_items}"

    return config, logger


def single_run_beat_the_pivot(
    set_of_items: SetofItems,
    sample_sizes: List[int],
    args: omegaconf.omegaconf.DictConfig,
    logger: logging.Logger = None,
):
    # args.final_sample_size will keep a count of the exact
    # number of times oracle will be called.
    args.final_sample_size = 0

    if args.color_aware:
        raise ValueError("Color-aware not relevant here")
    else:

        thetas = set_of_items.get_thetas()
        theta_biases = [item.theta_bias for item in set_of_items.items]

        # multiply thetas with theta_biases
        thetas = [theta * bias for theta, bias in zip(thetas, theta_biases)]

        pmodel = pairwise(len(thetas))
        pmodel.generate_deterministic_BTL_custom(thetas)
        kset = [val for val in range(1, len(thetas) + 1)]
        ar = ARalg(pmodel, kset)
        trackdata = ar.rank(args.dataset_config[args.dataset].delta, track=sample_sizes)

        predicted_rankings = []
        final_sample_sizes = []
        for item in trackdata:
            final_sample_sizes.append(item[0])
            predicted_rankings.append(item[1])

    return predicted_rankings, final_sample_sizes


def single_run_wrapper(args):
    # Wrapper function to call single_run_beat_the_pivot
    _set_seed(args["args"].seed)
    return single_run_beat_the_pivot(
        args["set_of_items"], args["sample_sizes"], args["args"], args["logger"]
    )


def post_process_ranking(ranking):
    if not isinstance(ranking, list):
        return None
    for rank in ranking:
        assert len(rank) == 1

    final_ranking = []
    for rank in ranking:
        final_ranking.append(f"item-{rank[0]+1}")
    return final_ranking


def multi_run_beat_the_pivot(
    set_of_items: SetofItems,
    sample_sizes: List[int],
    args: omegaconf.omegaconf.DictConfig,
    logger: logging.Logger = None,
):

    results = defaultdict()
    experiment_args = []
    for seed in range(1, args.num_of_exps + 1):

        exp_args = deepcopy(args)
        exp_args.seed = seed
        experiment_args.append(
            {
                "set_of_items": set_of_items,
                "sample_sizes": sample_sizes,
                "args": exp_args,
                "logger": logger,
            }
        )

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for predicted_ranking, final_sample_size in executor.map(
            single_run_wrapper, experiment_args
        ):
            for sample_size, ranking in zip(final_sample_size, predicted_ranking):
                if sample_size not in results:
                    results[sample_size] = defaultdict(list)
                ranking = post_process_ranking(ranking)
                if ranking is not None:
                    results[sample_size]["predicted_ranking"].append(ranking)
                    results[sample_size]["sample_size"].append(sample_size)

    return results


def main(
    args: omegaconf.dictconfig.DictConfig,
    logger: logging.Logger = None,
    experiment_folder: str = None,
    dataset_name: str = None,
    sample_sizes: List[int] = None,
):
    args.num_of_groups = getattr(
        args.dataset_config[args.dataset], "num_of_groups", None
    )
    # Create the ranking
    set_of_items = create_ranking(
        num_of_items=args.num_of_items,
        num_of_groups=args.num_of_groups,
        dataset=args.dataset,
        prot_attrs=(
            None
            if args.dataset in ["compas", "german"]
            else create_random_prot_attrs(args)
        ),
        args=args,
        logger=logger,
    )

    if experiment_folder is not None:
        # save the set_of_items in the experiment folder
        with open(f"{experiment_folder}/set_of_items_{dataset_name}.pkl", "wb") as f:
            pickle.dump(set_of_items, f)

    results = {}
    args.color_aware = False

    results["color-blind"] = multi_run_beat_the_pivot(
        set_of_items, sample_sizes, args, logger
    )
    results["set_of_items"] = set_of_items

    return results


import argparse
import numpy as np


def load_beat_the_pivot_results(exp_id):
    try:
        results, args, logger = pickle.load(
            open(f"./checkpoints/{exp_id}/beat_the_pivot_exp_50.pkl", "rb")
        )
    except FileNotFoundError:
        results, args, logger = pickle.load(
            open(f"./checkpoints/{exp_id}/baseline_50.pkl", "rb")
        )
    return results, args, logger


def get_btp_sample_sizes(btp_results, dataset):
    sample_sizes = [
        int(np.mean(val["sample_size"]))
        for k, val in btp_results[dataset]["color-blind"].items()
    ]
    return sample_sizes


if __name__ == "__main__":
    """
    sample command: python run_baseline.py
    """

    # first we need to load the corresponding beat_the_pivot results
    parser = argparse.ArgumentParser(description="Run baseline experiments.")
    parser.add_argument(
        "--exp-id", type=str, required=True, help="Experiment ID to load results for"
    )
    args = parser.parse_args()

    # Load the results using the provided exp_id
    btp_results, _, _ = load_beat_the_pivot_results(args.exp_id)

    # Load omegaconf config from a YAML file
    args, logger = read_config()
    logger.info("Running multirun experiment")

    # create a folder inside args.checkpoint.path with the latest date
    today_date_time = datetime.datetime.now().strftime("baseline_%Y-%m-%d_%H-%M-%S")
    experiment_folder = f"{args.checkpoint.path}/{today_date_time}"
    os.makedirs(experiment_folder, exist_ok=True)

    # save the config file in the experiment folder
    omegaconf.OmegaConf.save(args, f"{experiment_folder}/config_baseline.yaml")

    results = {}
    for args.dataset in args.datasets:

        if hasattr(args.dataset_config[args.dataset], "protected_group"):
            # assert that protected_group is a ListConf of OmegaConf
            assert isinstance(
                args.dataset_config[args.dataset].protected_group,
                omegaconf.listconfig.ListConfig,
            ), "protected_group must be a list"

            protected_groups = list(args.dataset_config[args.dataset].protected_group)
            # run the experiment for each protected_group of this dataset
            for prot_group in protected_groups:
                # set the current protected group for the dataset in args
                args.dataset_config[args.dataset].protected_group = prot_group
                print(f"\n\nRunning experiment for {args.dataset} with {prot_group}")
                # run the experiment

                btp_sample_sizes = get_btp_sample_sizes(
                    btp_results, f"{args.dataset}_{prot_group}"
                )

                results[f"{args.dataset}_{prot_group}"] = main(
                    args,
                    logger,
                    experiment_folder,
                    dataset_name=f"{args.dataset}_{prot_group}",
                    sample_sizes=btp_sample_sizes,
                )
        else:
            print(f"Running {args.dataset}")

            # get sample sizes from btp_results
            btp_sample_sizes = get_btp_sample_sizes(btp_results, args.dataset)

            results[args.dataset] = main(
                args,
                logger,
                experiment_folder,
                dataset_name=f"{args.dataset}",
                sample_sizes=btp_sample_sizes,
            )
        print("Got results")

    # pickle results and args
    with open(f"{experiment_folder}/baseline_{args.num_of_items}.pkl", "wb") as f:
        pickle.dump([results, args, logger], f)
