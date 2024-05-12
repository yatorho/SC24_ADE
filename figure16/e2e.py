import os
import warnings
from typing import Dict, List, cast

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from ecorec.costs.sharding_plan import ShardingMethod, estimate_cost_matrix
from ecorec.distributed import (batch_wise, data_parallel, model_parallel,
                                table_wise)
from ecorec.dlrm import DLRM, DLRM_Meta
from ecorec.models import MLP_Meta, TTEmbeddingLayer_Meta
from ecorec.utils import (DurationTimer, interpret_dist, nvtx_pop, nvtx_push,
                          r0_print)
from torch.cuda import cudart

from data.dlrm_ds_stats import DLRMDataset, keyed_dlrm_datasets_load


def match_sharding_type(sharding_method: str) -> table_wise.ShardingMethod:
    if sharding_method == "greed":
        return ShardingMethod.GREED
    elif sharding_method == "random":
        return ShardingMethod.RANDOM
    elif sharding_method == "sequential":
        return ShardingMethod.SEQUENTIAL
    else:
        raise ValueError(f"Invalid sharding method: {sharding_method}")


class DataParallel:
    def __init__(self):
        self.engine = data_parallel.DataParallelEngine

    def generate_plan(self, rank, global_keys, world_size, device):
        if rank == 0:
            plan, costs = self.engine.generate_plan(global_keys, world_size)
            plan_costs = [plan, costs]

        else:
            plan_costs = [None, None]

        dist.broadcast_object_list(plan_costs, src=0, device=device)
        plan, costs = plan_costs
        plan = cast(Dict[int, List[str]], plan)
        costs = cast(Dict[int, float], costs)

        local_keys = plan[rank]
        print(f"rank: {rank}, num_keys: {len(local_keys)}, local_keys: {local_keys}")
        r0_print(
            f"########### costs: {costs}, max: {max(costs.values()):.4f}, min: {min(costs.values()):.4f}"
        )

        return plan, costs, local_keys

    @staticmethod
    def generate_datasets(
        # sparse part
        global_keys: List[str],
        sp_dir: str,
        # dense part
        local_batch_size: int,
        dense_in_features: int,
        device: torch.device,
        iter_num: int = 1,
        verbose=False,
    ):
        feature_batches = keyed_dlrm_datasets_load(
            batch_size=local_batch_size,
            keys=global_keys,
            sp_dir=sp_dir,
            iter_num=iter_num,
            device=device,
            verbose=False,
        )

        dlrm_datasets = DLRMDataset(
            batch_size=local_batch_size,
            dense_in_features=dense_in_features,
            sparse=feature_batches,
            iter_num=iter_num,
            device=device,
        )

        return dlrm_datasets

    def run(self, local_rank, rank, world_size, args):
        device = torch.device(f"cuda:{local_rank}")

        local_batch_size = args.local_batch_size
        global_batch_size = local_batch_size * world_size
        embedding_dim = args.embedding_dim
        tt_ranks = args.tt_ranks
        tt_type = args.model
        num_global_keys = args.num_global_keys
        global_keys = [f"f{i}" for i in range(num_global_keys)]
        dense_in_features = args.dense_in_features
        dense_arch_layer_sizes = args.dense_arch_layer_sizes
        over_arch_layer_sizes = args.over_arch_layer_sizes
        sp_dir = args.sp_dir

        plan, costs, local_keys = self.generate_plan(
            rank, global_keys, world_size, device
        )

        # ================== for DLRM datasets =================
        dataset = DataParallel.generate_datasets(
            global_keys,
            sp_dir,
            local_batch_size,
            dense_in_features,
            device,
            iter_num=args.num_batches,
            verbose=args.verbose,
        )

        # ================== for DLRM components =================
        embedding_meta = TTEmbeddingLayer_Meta(
            batch_size=local_batch_size,
            keyed_num_embeddings=dataset.sparse.num_indices,
            embedding_dim=embedding_dim,
            tt_ranks=tt_ranks,
            keys=global_keys,
            tt_emb=tt_type,
            device=device,
            sharding_plan=plan,
            sharding_cost=costs,
            learning_rate=args.learning_rate,
        )
        mlps_meta = MLP_Meta(
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=dense_arch_layer_sizes,
            num_sparse_features=num_global_keys,
            over_arch_layer_sizes=over_arch_layer_sizes,
            embedding_dim=embedding_dim,
            device=device,
        )

        dlrm_meta = DLRM_Meta(embedding_meta, mlps_meta)
        dlrm = DLRM(dlrm_meta).to(device)

        # ================== for DLRM training =================
        iters: int = args.iters
        runtime_ctx = data_parallel.RuntimeContext(
            global_keys,
            embedding_dim,
            False,
            device,
            torch.zeros(local_batch_size, embedding_dim, device=device),
            nn.MSELoss(),
            init_grad=torch.randn(
                local_batch_size, over_arch_layer_sizes[-1], device=device
            )
            * 1e-6,
        )

        nvtx_push("warmup")
        for i in range(iters):
            for batch in dataset:
                self.engine.execute_batch(
                    batch, dlrm.embeddings, dlrm.mlps, runtime_ctx
                )

        torch.cuda.synchronize(device)
        dist.barrier()
        nvtx_pop()

        runtime_ctx.nvtx_cond = True
        with DurationTimer() as t:
            for i in range(iters):
                nvtx_push(f"i: {i}")
                for batch in dataset:
                    self.engine.execute_batch(
                        batch, dlrm.embeddings, dlrm.mlps, runtime_ctx
                    )
                nvtx_pop()

        duration = t.get_duration() / iters

        return duration


class ModelParallel:
    def __init__(
        self,
    ):
        self.engine = model_parallel.ModelParallelEngine

    def generate_plan(self, rank, local_rank, args):
        device = torch.device(f"cuda:{local_rank}")
        global_keys = [f"f{i}" for i in range(args.num_global_keys)]

        sp_dir = args.sp_dir
        # ================== for sharding plan =================
        if rank == 0:
            eval_batch_size = 131072
            sharding_method = match_sharding_type(args.sharding_method)

            cost_matrix = estimate_cost_matrix(
                keyed_dlrm_datasets_load(
                    batch_size=eval_batch_size,
                    keys=global_keys,
                    sp_dir=sp_dir,
                    iter_num=1,
                    device=device,
                ),
                keys=global_keys,
                verbose=args.verbose,
            )

            plan, costs = self.engine.generate_plan(cost_matrix, sharding_method)
            plan_costs = [plan, costs]

            del cost_matrix
        else:
            plan_costs = [None, None]

        dist.broadcast_object_list(plan_costs, src=0, device=device)
        plan, costs = plan_costs
        plan = cast(Dict[int, List[str]], plan)
        costs = cast(Dict[int, float], costs)

        local_keys = plan[rank]
        print(f"rank: {rank}, num_keys: {len(local_keys)}, local_keys: {local_keys}")
        r0_print(
            f"########### costs: {costs}, max: {max(costs.values()):.4f}, min: {min(costs.values()):.4f}"
        )

        return plan, costs, local_keys

    @staticmethod
    def generate_datasets(
        # sparse part
        global_batch_size: int,
        local_keys: List[str],
        sp_dir: str,
        # dense part
        local_batch_size: int,
        dense_in_features: int,
        device: torch.device,
        iter_num: int = 1,
        verbose=False,
    ):
        feature_batches = keyed_dlrm_datasets_load(
            batch_size=global_batch_size,
            keys=local_keys,
            sp_dir=sp_dir,
            iter_num=iter_num,
            device=device,
            verbose=verbose,
        )

        dlrm_datasets = DLRMDataset(
            batch_size=local_batch_size,
            dense_in_features=dense_in_features,
            sparse=feature_batches,
            iter_num=iter_num,
            device=device,
        )

        return dlrm_datasets

    def run(self, local_rank, rank, world_size, args):
        device = torch.device(f"cuda:{local_rank}")

        local_batch_size = args.local_batch_size
        global_batch_size = local_batch_size * world_size
        embedding_dim = args.embedding_dim
        tt_ranks = args.tt_ranks
        tt_type = args.model
        num_global_keys = args.num_global_keys
        global_keys = [f"f{i}" for i in range(num_global_keys)]
        dense_in_features = args.dense_in_features
        dense_arch_layer_sizes = args.dense_arch_layer_sizes
        over_arch_layer_sizes = args.over_arch_layer_sizes
        sp_dir = args.sp_dir

        plan, costs, local_keys = self.generate_plan(rank, local_rank, args)

        # ================== for DLRM datasets =================
        dataset = ModelParallel.generate_datasets(
            global_batch_size,
            local_keys,
            sp_dir,
            local_batch_size,
            dense_in_features,
            device,
            iter_num=args.num_batches,
            verbose=args.verbose,
        )

        # ================== for DLRM components =================
        embedding_meta = TTEmbeddingLayer_Meta(
            batch_size=local_batch_size,
            keyed_num_embeddings=dataset.sparse.num_indices,
            embedding_dim=embedding_dim,
            tt_ranks=tt_ranks,
            keys=global_keys,
            tt_emb=tt_type,
            device=device,
            sharding_plan=plan,
            sharding_cost=costs,
            learning_rate=args.learning_rate,
        )
        mlps_meta = MLP_Meta(
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=dense_arch_layer_sizes,
            num_sparse_features=num_global_keys,
            over_arch_layer_sizes=over_arch_layer_sizes,
            embedding_dim=embedding_dim,
            device=device,
        )

        dlrm_meta = DLRM_Meta(embedding_meta, mlps_meta)
        dlrm = DLRM(dlrm_meta).to(device)

        # ================== for DLRM training =================
        iters: int = args.iters
        runtime_ctx = model_parallel.RuntimeContext(
            local_keys,
            global_keys,
            embedding_dim,
            plan,
            False,
            device,
            torch.zeros(global_batch_size, embedding_dim, device=device),
            nn.MSELoss(),
            args.check_result,
            init_grad=torch.randn(
                local_batch_size, over_arch_layer_sizes[-1], device=device
            )
            * 1e-6,
        )

        nvtx_push("warmup")
        for i in range(iters):
            for batch in dataset:
                self.engine.execute_batch(
                    batch, dlrm.embeddings, dlrm.mlps, runtime_ctx
                )

        torch.cuda.synchronize(device)
        dist.barrier()
        nvtx_pop()

        runtime_ctx.nvtx_cond = True
        with DurationTimer() as t:
            for i in range(iters):
                nvtx_push(f"i: {i}")
                for batch in dataset:
                    self.engine.execute_batch(
                        batch, dlrm.embeddings, dlrm.mlps, runtime_ctx
                    )
                nvtx_pop()

        duration = t.get_duration() / iters

        return duration


class BatchWisePipeline:
    def __init__(
        self,
    ):
        self.engine = batch_wise.BatchWisePipelineEngine

    def generate_plan(self, rank, local_rank, args):
        device = torch.device(f"cuda:{local_rank}")
        global_keys = [f"f{i}" for i in range(args.num_global_keys)]

        sp_dir = args.sp_dir
        # ================== for sharding plan =================
        if rank == 0:
            eval_batch_size = 131072
            sharding_method = match_sharding_type(args.sharding_method)

            cost_matrix = estimate_cost_matrix(
                keyed_dlrm_datasets_load(
                    batch_size=eval_batch_size,
                    keys=global_keys,
                    sp_dir=sp_dir,
                    iter_num=1,
                    device=device,
                ),
                keys=global_keys,
                verbose=args.verbose,
            )

            plan, costs = self.engine.generate_plan(cost_matrix, sharding_method)
            plan_costs = [plan, costs]

            del cost_matrix
        else:
            plan_costs = [None, None]

        dist.broadcast_object_list(plan_costs, src=0, device=device)
        plan, costs = plan_costs
        plan = cast(Dict[int, List[str]], plan)
        costs = cast(Dict[int, float], costs)

        local_keys = plan[rank]
        print(f"rank: {rank}, num_keys: {len(local_keys)}, local_keys: {local_keys}")
        r0_print(
            f"########### costs: {costs}, max: {max(costs.values()):.4f}, min: {min(costs.values()):.4f}"
        )

        return plan, costs, local_keys
    
    @staticmethod
    def generate_datasets(
        # sparse part
        global_batch_size: int,
        local_keys: List[str],
        sp_dir: str,
        # dense part
        local_batch_size: int,
        dense_in_features: int,
        device: torch.device,
        iter_num: int = 1,
        verbose=False,
    ):
        feature_batches = keyed_dlrm_datasets_load(
            batch_size=global_batch_size,
            keys=local_keys,
            sp_dir=sp_dir,
            iter_num=iter_num,
            device=device,
            verbose=verbose,
        )

        dlrm_datasets = DLRMDataset(
            batch_size=local_batch_size,
            dense_in_features=dense_in_features,
            sparse=feature_batches,
            iter_num=iter_num,
            device=device,
        )

        return dlrm_datasets

    def run(self, local_rank, rank, world_size, args):
        device = torch.device(f"cuda:{local_rank}")

        local_batch_size = args.local_batch_size
        global_batch_size = local_batch_size * world_size
        embedding_dim = args.embedding_dim
        tt_ranks = args.tt_ranks
        tt_type = args.model
        num_global_keys = args.num_global_keys
        global_keys = [f"f{i}" for i in range(num_global_keys)]
        dense_in_features = args.dense_in_features
        dense_arch_layer_sizes = args.dense_arch_layer_sizes
        over_arch_layer_sizes = args.over_arch_layer_sizes
        sp_dir = args.sp_dir

        plan, costs, local_keys = self.generate_plan(rank, local_rank, args)

        # ================== for DLRM datasets =================
        dataset = ModelParallel.generate_datasets(
            global_batch_size,
            local_keys,
            sp_dir,
            local_batch_size,
            dense_in_features,
            device,
            iter_num=args.num_batches,
            verbose=args.verbose,
        )

        # ================== for DLRM components =================
        embedding_meta = TTEmbeddingLayer_Meta(
            batch_size=local_batch_size,
            keyed_num_embeddings=dataset.sparse.num_indices,
            embedding_dim=embedding_dim,
            tt_ranks=tt_ranks,
            keys=global_keys,
            tt_emb=tt_type,
            device=device,
            sharding_plan=plan,
            sharding_cost=costs,
            learning_rate=args.learning_rate,
        )
        mlps_meta = MLP_Meta(
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=dense_arch_layer_sizes,
            num_sparse_features=num_global_keys,
            over_arch_layer_sizes=over_arch_layer_sizes,
            embedding_dim=embedding_dim,
            device=device,
        )

        dlrm_meta = DLRM_Meta(embedding_meta, mlps_meta)
        dlrm = DLRM(dlrm_meta).to(device)

        # ================== for DLRM training =================
        iters = args.iters
        comp_stream = torch.cuda.current_stream(device=device)  # for computing
        comm_stream: torch.cuda.Stream = torch.cuda.Stream(device=device)  # type: ignore # for communication

        runtime_ctx = batch_wise.RuntimeContext(
            rank,
            local_keys,
            global_keys,
            global_batch_size,
            local_batch_size,
            embedding_dim,
            plan,
            False,
            comp_stream,
            comm_stream,
            device,
            torch.zeros(global_batch_size, embedding_dim, device=device),
            nn.MSELoss(),
            args.num_micro_batches,
            args.check_result,

            init_grad=torch.randn(
                local_batch_size, over_arch_layer_sizes[-1], device=device
            ) * 1e-6,
        )

        nvtx_push("warmup")
        for i in range(iters):
            for batch in dataset:
                self.engine.execute_batch(
                    batch, dlrm.embeddings, dlrm.mlps, runtime_ctx
                )

        torch.cuda.synchronize(device)
        dist.barrier()
        nvtx_pop()

        runtime_ctx.nvtx_cond = True
        with DurationTimer() as t:
            for i in range(iters):
                nvtx_push(f"i: {i}")
                for batch in dataset:
                    self.engine.execute_batch(
                        batch, dlrm.embeddings, dlrm.mlps, runtime_ctx
                    )
                nvtx_pop()

        duration = t.get_duration() / iters

        return duration




class TableWisePipeline:
    def __init__(
        self,
    ):
        self.engine = table_wise.TableWisePipelineEngine

    def generate_plan(self, rank, local_rank, args):
        device = torch.device(f"cuda:{local_rank}")
        global_keys = [f"f{i}" for i in range(args.num_global_keys)]

        sp_dir = args.sp_dir
        # ================== for sharding plan =================
        if rank == 0:
            eval_batch_size = args.local_batch_size
            sharding_method = match_sharding_type(args.sharding_method)
            reordering = args.reordering

            cost_matrix = estimate_cost_matrix(
                keyed_dlrm_datasets_load(
                    batch_size=eval_batch_size,
                    keys=global_keys,
                    sp_dir=sp_dir,
                    iter_num=1,
                    device=device,
                ),
                keys=global_keys,
                verbose=args.verbose,
            )

            plan, costs = self.engine.generate_plan(
                cost_matrix, reordering, sharding_method
            )
            plan_costs = [plan, costs]

            del cost_matrix
        else:
            plan_costs = [None, None]

        dist.broadcast_object_list(plan_costs, src=0, device=device)
        plan, costs = plan_costs
        plan = cast(Dict[int, List[str]], plan)
        costs = cast(Dict[int, float], costs)

        local_keys = plan[rank]
        print(f"rank: {rank}, num_keys: {len(local_keys)}, local_keys: {local_keys}")
        r0_print(
            f"########### costs: {costs}, max: {max(costs.values()):.4f}, min: {min(costs.values()):.4f}"
        )

        min_shard_features = min([len(plan[r]) for r in plan.keys()])
        if min_shard_features < args.num_micro_keys:
            warnings.warn(
                f"min_shard_features({min_shard_features}) < num_micro_keys({args.num_micro_keys}). We will set num_micro_keys to {min_shard_features}"
            )

            args.num_micro_keys = min_shard_features

        return plan, costs, local_keys

    @staticmethod
    def generate_datasets(
        # sparse part
        global_batch_size: int,
        local_keys: List[str],
        sp_dir: str,
        # dense part
        local_batch_size: int,
        dense_in_features: int,
        device: torch.device,
        iter_num: int = 1,
        verbose=False,
    ):
        feature_batches = keyed_dlrm_datasets_load(
            batch_size=global_batch_size,
            keys=local_keys,
            sp_dir=sp_dir,
            iter_num=iter_num,
            device=device,
            verbose=False,
        )

        dlrm_datasets = DLRMDataset(
            batch_size=local_batch_size,
            dense_in_features=dense_in_features,
            sparse=feature_batches,
            iter_num=iter_num,
            device=device,
        )

        return dlrm_datasets

    def run(self, local_rank, rank, world_size, args):
        device = torch.device(f"cuda:{local_rank}")

        local_batch_size = args.local_batch_size
        global_batch_size = local_batch_size * world_size
        embedding_dim = args.embedding_dim
        tt_ranks = args.tt_ranks
        tt_type = args.model
        num_global_keys = args.num_global_keys
        global_keys = [f"f{i}" for i in range(num_global_keys)]
        dense_in_features = args.dense_in_features
        dense_arch_layer_sizes = args.dense_arch_layer_sizes
        over_arch_layer_sizes = args.over_arch_layer_sizes
        sp_dir = args.sp_dir

        plan, costs, local_keys = self.generate_plan(rank, local_rank, args)

        # ================== for DLRM datasets =================
        dataset = TableWisePipeline.generate_datasets(
            global_batch_size,
            local_keys,
            sp_dir,
            local_batch_size,
            dense_in_features,
            device,
            iter_num=args.num_batches,
            verbose=args.verbose,
        )

        # ================== for DLRM components =================
        embedding_meta = TTEmbeddingLayer_Meta(
            batch_size=local_batch_size,
            keyed_num_embeddings=dataset.sparse.num_indices,
            embedding_dim=embedding_dim,
            tt_ranks=tt_ranks,
            keys=global_keys,
            tt_emb=tt_type,
            device=device,
            sharding_plan=plan,
            sharding_cost=costs,
            learning_rate=args.learning_rate,
        )
        mlps_meta = MLP_Meta(
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=dense_arch_layer_sizes,
            num_sparse_features=num_global_keys,
            over_arch_layer_sizes=over_arch_layer_sizes,
            embedding_dim=embedding_dim,
            device=device,
        )

        dlrm_meta = DLRM_Meta(embedding_meta, mlps_meta)
        dlrm = DLRM(dlrm_meta).to(device)

        # ================== for DLRM training =================
        comp_stream = torch.cuda.current_stream(device=device)  # for computing
        comm_stream: torch.cuda.Stream = torch.cuda.Stream(device=device)  # type: ignore # for communication

        num_micro_keys = args.num_micro_keys
        num_micro_uidx = args.num_micro_uidx
        iters: int = args.iters

        runtime_ctx = table_wise.RuntimeContext(
            rank,
            global_keys,
            local_batch_size,
            embedding_dim,
            plan,
            False,
            comp_stream,
            comm_stream,
            device,
            torch.zeros(global_batch_size, embedding_dim, device=device),
            nn.MSELoss(),
            num_micro_keys,
            num_micro_uidx,
            args.skew_degree,
            args.check_result,
            init_grad=torch.randn(
                local_batch_size, over_arch_layer_sizes[-1], device=device
            )
            * 1e-6,
        )

        nvtx_push("warmup")
        for i in range(iters):
            for batch in dataset:
                self.engine.execute_batch(
                    batch, dlrm.embeddings, dlrm.mlps, runtime_ctx
                )

        torch.cuda.synchronize(device)
        dist.barrier()
        nvtx_pop()

        runtime_ctx.nvtx_cond = True
        with DurationTimer() as t:
            for i in range(iters):
                nvtx_push(f"i: {i}")
                for batch in dataset:
                    self.engine.execute_batch(
                        batch, dlrm.embeddings, dlrm.mlps, runtime_ctx
                    )
                nvtx_pop()

        duration = t.get_duration() / iters

        return duration


def reduce_number(num: float, device, rank=0):
    num_t = torch.tensor(num, device=device)
    dist.all_reduce(num_t, op=dist.ReduceOp.SUM)
    return num_t.item() / dist.get_world_size()


def run(local_rank, rank, world_size, args):

    cudart().cudaProfilerStart()  # type: ignore
    dist.barrier()

    parallel = args.parallel

    if parallel == "dp":
        impl = DataParallel()
        duration = impl.run(local_rank, rank, world_size, args)
    elif parallel == "mp":
        impl = ModelParallel()
        duration = impl.run(local_rank, rank, world_size, args)
    elif parallel == "bwp":
        impl = BatchWisePipeline()
        duration = impl.run(local_rank, rank, world_size, args)
    elif parallel == "twp":
        impl = TableWisePipeline()
        duration = impl.run(local_rank, rank, world_size, args)
    else:
        raise ValueError(f"Not supported parallel: {parallel}")

    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    print("rank: {}, peak memory: {}MB, time: {}ms".format(rank, peak_mem, duration))

    device = torch.device(f"cuda:{local_rank}")
    duration = reduce_number(duration, device, rank)
    peak_mem = reduce_number(peak_mem, device, rank)
    if rank == 0:
        with open(args.log_file, "w") as f:
            f.write(f"peak memory: {peak_mem:.4f}MB, time_per_iter: {duration:.4f}ms\n")

    # process sync
    dist.barrier()
    cudart().cudaProfilerStop()  # type: ignore


def init_dist(world_size, rank, master_addr, master_port, local_rank, args):
    mp.set_start_method("spawn")

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)

    print_args(args)
    run(local_rank, rank, world_size, args)


def print_args(args):
    r0_print("Arguments:")
    for arg in vars(args):
        r0_print(f"\t{arg}: {getattr(args, arg)}")


if __name__ == "__main__":
    def _parse_comma_separated_list(s):
        return list(map(int, s.split(",")))

    import argparse

    sp_dir = "datasets/dlrm_pt/2022/splits"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--launch", type=str, default="torch", choices=["torch", "slurm"]
    )
    parser.add_argument(
        "--parallel", type=str, choices=["dp", "mp", "twp", "bwp"], default="dp"
    )
    parser.add_argument("--num_global_keys", type=int, default=100)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--tt_ranks", type=_parse_comma_separated_list, default="32,32")
    parser.add_argument("--local_batch_size", type=int, default=131072)
    parser.add_argument(
        "--sharding_method",
        type=str,
        choices=["greed", "random", "sequential"],
        help="Balanced TT-EMBs sharding algorithm. Default: greed.",
        default="greed",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="EcoRec",
        choices=["EcoRec", "FBTT", "ELRec"],
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
    )
    parser.add_argument("--sp_dir", type=str, default=sp_dir)
    parser.add_argument(
        "--num_micro_keys",
        type=int,
        default=2,
        help="Number of stages in pipeline scheduling. This param only works with table-wise pipeline model.",
    )
    parser.add_argument(
        "--num_micro_batches",
        type=int,
        default=2,
        help="Number of stages in pipeline scheduling. This param only works with batch-wise pipeline model.",
    )
    parser.add_argument(
        "--num_micro_uidx",
        type=int,
        default=2,
        help="Number of micro-batches for micro-batching strategy. This param only works with table-wise pipeline model",
    )
    parser.add_argument(
        "--reordering",
        action="store_true",
        default=False,
        help="Whether to enable reordering feature strategy  for table-wise pipeline model.",
    )
    parser.add_argument(
        "--skew_degree",
        type=float,
        default=0.0,
        help="Skew degree for table-wise pipeline grain sharding, i.e., slope"
        "feature count strategy . This param only works with table-wise pipeline model.",
    )

    # ================== for MLP components =================
    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=_parse_comma_separated_list,
        default="256,512,256,32",
    )
    parser.add_argument("--dense_in_features", type=int, default=13)
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=_parse_comma_separated_list,
        default="512,256,128,1",
    )

    parser.add_argument("--check_result", action="store_true", default=False)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="print verbose"
    )
    parser.add_argument(
        "--log_file", type=str, default="result.txt", help="log file path"
    )

    args = parser.parse_args()

    init_dist(*interpret_dist(args), args)
