import os
import re
import sys
from multiprocessing import Pool

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from ecorec.models import create_embedding_bag

from data.dlrm_ds_stats import keyed_dlrm_datasets_load, ring_dataset


def cond_profile(
    model, key, device, tt_ranks, embedding_dim, batch_size, sp_dir, warmup, iters
):
    torch.cuda.set_device(device)

    datasets = keyed_dlrm_datasets_load(
        batch_size=batch_size,
        keys=[key],
        sp_dir=sp_dir,
        iter_num=-1,  # max_iter_num
        verbose=False,
    )

    num_indices = datasets.num_indices[key]  # not reliable for batch.

    if num_indices <= 0:
        print("Skip empty dataset.")
        return 0, 0

    emb = create_embedding_bag(
        name=model,
        num_embeddings=num_indices,
        embedding_dim=embedding_dim,
        tt_ranks=tt_ranks,
        learning_rate=0.01,
        device=device,
    )

    def func():
        events = []

        for batch in ring_dataset(
            datasets,
            num_iters=warmup + iters,
        ):
            indices = batch[key].values().to(device)
            offsets = batch[key].offsets().to(device)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            if indices.numel() > 0:
                emb(indices, offsets)
            end_event.record()

            events.append((start_event, end_event))

        return events

    for _ in range(warmup):
        func()
    torch.cuda.synchronize(device)

    events = []
    for _ in range(iters):
        es = func()
        events.extend(es)
    torch.cuda.synchronize(device)

    time = 0
    for start_event, end_event in events:
        time += start_event.elapsed_time(end_event)

    time /= iters * len(datasets)
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

    return time, peak_mem


def clear_torch_extension_cache():
    root_extensions_directory = os.environ.get("TORCH_EXTENSIONS_DIR")
    if root_extensions_directory is None:
        root_extensions_directory = os.path.realpath(
            torch._appdirs.user_cache_dir(appname="torch_extensions")
        )
        cu_str = (
            "cpu"
            if torch.version.cuda is None
            else f'cu{torch.version.cuda.replace(".", "")}'
        )  # type: ignore[attr-defined]
        python_version = f"py{sys.version_info.major}{sys.version_info.minor}"
        build_folder = f"{python_version}_{cu_str}"

        root_extensions_directory = os.path.join(
            root_extensions_directory, build_folder
        )
    ecorec_extension_name = "ecorec_kernel"


def main(
    world_size,
    rank,
    local_rank,
    sp_dir,
    cost_dir,
    batch_size,
    embedding_dim,
    tt_ranks,
    num_features,
    models,
    warmup,
    iters,
):
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    keys = [f"f{i}" for i in range(num_features)]

    # This is a bug that torch.cpp_extension could not load ecorec extension after using `multiprocessing.Pool.`
    # So, it is a compromise to remove the cache of ecorec extension build file.
    clear_torch_extension_cache()
    costs_path = os.path.join(
        costs_dir,
        f"costs_f{num_features}_b{batch_size}_r{tt_ranks}_v{embedding_dim}_d{local_rank}_gr{rank}.pt",
    )

    if not os.path.exists(costs_dir):
        os.makedirs(costs_dir)

    costs = {}

    for model in models:
        for key in keys:
            with Pool(1) as pool:
                print(f"rank: {rank} Model: {model}, Key: {key}, Device: {device}")
                res = pool.apply(
                    cond_profile,
                    kwds=dict(
                        model=model,
                        key=key,
                        device=device,
                        tt_ranks=tt_ranks,
                        embedding_dim=embedding_dim,
                        batch_size=batch_size,
                        sp_dir=sp_dir,
                        warmup=warmup,
                        iters=iters,
                    ),
                )
                print(f"rank:{rank} {res}")

                cost_key = str(
                    (model, key, device, tt_ranks, embedding_dim, batch_size)
                )
                cost_value = res

                costs[cost_key] = cost_value

                torch.save(costs, costs_path[:-3] + "_update.pt")

    print(f"rank:{rank}: ++++++++++++++++++++++++++++++++++++++++")
    print(costs)
    torch.save(costs, costs_path)

    # remove update file
    os.remove(costs_path[:-3] + "_update.pt")


def interpret_dist(args):
    launch_method = args.launch

    if launch_method == "torch":
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]
    elif launch_method == "slurm":
        local_rank = int(os.environ["SLURM_LOCALID"])
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLUM_NNODES"]) * int(
            os.environ["SLUM_NTASKS_PER_NODE"]
        )
        master_addr = os.environ["SLUM_SUBMIT_HOST"]
        master_port = "1234"
    else:
        raise NotImplementedError

    return world_size, rank, master_addr, master_port, local_rank


def run(world_size, rank, master_addr, master_port, local_rank, args):
    mp.set_start_method("spawn")

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    
    sp_dir = args.sp_dir
    costs_dir = args.costs_dir
    batch_size = args.batch_size
    embedding_dim = args.embedding_dim
    tt_ranks = args.tt_ranks
    num_features = args.num_features
    models = args.models
    warmup = args.warmup
    iters = args.iters

    main(
        world_size,
        rank,
        local_rank,
        sp_dir=sp_dir,
        cost_dir=costs_dir,
        batch_size=batch_size,
        embedding_dim=embedding_dim,
        tt_ranks=tt_ranks,
        num_features=num_features,
        models=models,
        warmup=warmup,
        iters=iters,
    )


if __name__ == "__main__":
    import argparse

    def _parse_comma_separated_list(s):
        return list(map(int, s.split(",")))

    sp_dir = "/home/yatorho/Documents/Code/TT/tt_dlrm/dlrm_v1/embeddingv1/dlrm_pt/2022/splits"
    costs_dir = (
        "/home/yatorho/Documents/Code/TT/tt_dlrm/dlrm_v1/embeddingv1/cost_stats/"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", type=str, default="torch")

    parser.add_argument("--sp_dir", type=str, default=sp_dir)
    parser.add_argument("--costs_dir", type=str, default=costs_dir)
    parser.add_argument("--batch_size", type=int, default=131072)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--tt_ranks", type=_parse_comma_separated_list, default="32,32")
    parser.add_argument("--num_features", type=int, default=100)
    parser.add_argument(
        "--models", type=lambda s: s.split(","), default="FBTT,ELRec,EcoRec"
    )
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--iters", type=int, default=15)

    args = parser.parse_args()

    run(*interpret_dist(args), args)
