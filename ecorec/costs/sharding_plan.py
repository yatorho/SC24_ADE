import enum
import os
from cmath import cos
from multiprocessing import Pool
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.cuda import cudart, nvtx
from torchrec import KeyedJaggedTensor


class ShardingMethod(enum.Enum):
    GREED = "greed"
    RANDOM = "random"
    SEQUENTIAL = "sequential"


KeyedCostMatrix = Dict[str, Dict[int, float]]
ShardingPlan = Dict[int, List[str]]
ShardingCost = Dict[int, float]
PartitionFunc = Callable[[KeyedCostMatrix], Tuple[ShardingPlan, ShardingCost]]


def greed_partition(keyed_costs_matrix: Dict[str, Dict[int, float]]):
    costs_matrix = torch.tensor([list(v.values()) for v in keyed_costs_matrix.values()])
    keys = list(keyed_costs_matrix.keys())
    ranks = list(keyed_costs_matrix[keys[0]].keys())

    norm_costs = costs_matrix.sum(dim=1)  # (num_keys, )
    # sort the keys by their costs
    sorted_keys = torch.argsort(norm_costs, descending=True)

    sharding_plan: Dict[int, List[str]] = {r: [] for r in ranks}
    sharding_cost: Dict[int, float] = {r: 0.0 for r in ranks}

    for t_key in sorted_keys:
        i = int(t_key.item())
        key = keys[i]

        sharding_cost_compare = {
            shard: sharding_cost[shard] + keyed_costs_matrix[key][shard]
            for shard in ranks
        }
        shard = min(sharding_cost_compare, key=lambda x: sharding_cost_compare[x])

        sharding_plan[shard].append(key)
        sharding_cost[shard] += keyed_costs_matrix[key][shard]

    return sharding_plan, sharding_cost


def random_partition(keyed_costs_matrix: Dict[str, Dict[int, float]]):
    keys = list(keyed_costs_matrix.keys())
    ranks = list(keyed_costs_matrix[keys[0]].keys())

    sharding_plan: Dict[int, List[str]] = {r: [] for r in ranks}
    sharding_cost: Dict[int, float] = {r: 0.0 for r in ranks}

    for key in keys:
        # find the shard with the least cost
        shard = np.random.choice(ranks)
        sharding_plan[shard].append(key)
        sharding_cost[shard] += keyed_costs_matrix[key][shard]

    return sharding_plan, sharding_cost


def sequential_partition(keyed_costs_matrix: Dict[str, Dict[int, float]]):
    keys = list(keyed_costs_matrix.keys())
    ranks = list(keyed_costs_matrix[keys[0]].keys())

    sharding_plan: Dict[int, List[str]] = {r: [] for r in ranks}
    sharding_cost: Dict[int, float] = {r: 0.0 for r in ranks}

    for i, key in enumerate(keys):
        # find the shard with the least cost
        shard = ranks[i % len(ranks)]
        sharding_plan[shard].append(key)
        sharding_cost[shard] += keyed_costs_matrix[key][shard]

    return sharding_plan, sharding_cost


def _parse_cost_stats(cost_stats):
    keyed_costs = {}
    keyed_peak_mem = {}

    # keys = cost_stats.keys()
    for key in cost_stats.keys():
        model = key.split("'")[1]
        feature = key.split("'")[3]

        if model not in keyed_costs:
            keyed_costs[model] = {}
        if model not in keyed_peak_mem:
            keyed_peak_mem[model] = {}

        keyed_costs[model][feature] = cost_stats[key][0]
        keyed_peak_mem[model][feature] = cost_stats[key][1]

    return keyed_costs, keyed_peak_mem


def load_cost_stats(dir, need_batch_size, ranks=None, verbose=False):
    import re

    pattern = re.compile(r"costs_f(\d+)_b(\d+)_r\[([^]]+)\]_v(\d+)_d(\d+)_gr(\d+).pt")

    rank_keyed_costs = {}
    rank_set = set()

    for f in os.listdir(dir):
        match = pattern.match(f)
        if match:
            num_features = int(match.group(1))
            batch_size = int(match.group(2))
            tt_ranks = list(map(int, match.group(3).split(",")))
            embedding_dim = int(match.group(4))
            local_rank = int(match.group(5))
            rank = int(match.group(6))

            cost_stats = torch.load(os.path.join(dir, f))

            if (ranks is None or rank in ranks) and batch_size == need_batch_size:
                print(f"match requirement costs file: {f}") if verbose else None

                rank_set.add(rank)
                if rank not in rank_keyed_costs:
                    rank_keyed_costs[rank] = {}

                rank_keyed_costs[rank]["num_features"] = num_features
                rank_keyed_costs[rank]["batch_size"] = batch_size
                rank_keyed_costs[rank]["tt_ranks"] = tt_ranks
                rank_keyed_costs[rank]["embedding_dim"] = embedding_dim
                rank_keyed_costs[rank]["local_rank"] = local_rank

                keyed_costs, keyed_peak_mem = _parse_cost_stats(cost_stats)
                rank_keyed_costs[rank]["costs"] = keyed_costs
                rank_keyed_costs[rank]["peak_mem"] = keyed_peak_mem

    if ranks is not None and rank_set != set(ranks):
        raise ValueError(f"ranks {ranks} not found in {rank_set}")

    return rank_keyed_costs


def load_costs_matrix(
    rank_keyed_costs, model, keys=None
) -> Dict[str, Dict[int, float]]:
    models = [list(v["costs"].keys()) for v in rank_keyed_costs.values()]
    uini_models = set([item for sublist in models for item in sublist])
    for sublist in models:
        for model in sublist:
            assert model in uini_models

    assert model in uini_models, f"model {model} not found in {uini_models}"

    keyed_cost_matrix = {}
    key_set = set()

    for rank, rank_costs in rank_keyed_costs.items():
        # costs
        keyed_costs = rank_costs["costs"][model]

        for key, cost in keyed_costs.items():
            if keys is None or key in keys:
                key_set.add(key)

                if key not in keyed_cost_matrix:
                    keyed_cost_matrix[key] = {}

                keyed_cost_matrix[key][rank] = cost

    if keys is not None and key_set != set(keys):
        raise ValueError(f"keys {keys} not found in {key_set}")

    return keyed_cost_matrix


def estimate_cost_matrix(
    keyed_sparse_dataset,
    keys: Optional[List[str]] = None,
    verbose: bool = False,
):
    print("just estimate sharding plan") if verbose else None
    if keyed_sparse_dataset is None:
        raise ValueError("keyed_sparse_dataset is None")

    import torch.distributed as dist

    if dist.is_initialized():
        ranks = list(range(dist.get_world_size()))
    else:
        ranks = [0]

    if keys is not None:
        for key in keys:
            if key not in keyed_sparse_dataset.keys:  # type: ignore
                raise ValueError(f"key {key} not found in keyed_sparse_dataset")
    else:
        keys = list(keyed_sparse_dataset.keys)  # type: ignore

    # Assuming Ei is i-th stage multiplications count in C-M/M-C TT pattern.
    # FWD: E1 + E2
    # BWD: 3 * E1 + 2 * E2
    # Cause E1 << E2, we simply use E2, i.e., the unique number, to estimate the cost.
    keyed_costs_matrix: Dict[str, Dict[int, float]] = {}
    estimate_iters = 1
    for key in keys:
        num_uni = 0
        cur_iters = 0
        for kjt in keyed_sparse_dataset:
            jt = kjt[key]
            num_uni += torch.unique(jt.values()).numel()

            cur_iters += 1
            if cur_iters >= estimate_iters:
                break

        num_uni /= estimate_iters
        keyed_costs_matrix[key] = {rank: num_uni for rank in ranks}

    return keyed_costs_matrix


def sharding(
    keyed_costs_matrix: KeyedCostMatrix,
    partition: Union[
        ShardingMethod,
        PartitionFunc,
    ] = ShardingMethod.GREED,  # "greed", "random", "sequential", or a callable function
) -> Tuple[ShardingPlan, ShardingCost]:
    # keyed_costs_matrix = load_costs_matrix(
    #     load_cost_stats(cost_dir, batch_size, ranks, verbose),
    #     model=model,
    #     keys=keys,
    # )

    if partition ==  ShardingMethod.GREED:
        partial_func = greed_partition
    elif partition == ShardingMethod.RANDOM:
        partial_func = random_partition
    elif partition == ShardingMethod.SEQUENTIAL:
        partial_func = sequential_partition
    else:
        if callable(partition):
            partial_func = partition
        else:
            raise ValueError(f"Unknown partition: {partition}")

    sharding_plan, sharding_cost = partial_func(keyed_costs_matrix)

    return sharding_plan, sharding_cost
