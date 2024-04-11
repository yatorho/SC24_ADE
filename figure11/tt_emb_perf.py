import argparse

import torch

from data.dlrm_ds_stats import keyed_dlrm_datasets_load
from ecorec.models import create_embedding_bag
from ecorec.utils import DurationTimer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=131072)
    parser.add_argument(
        "--model", type=str, choices=["FBTT", "ELRec", "EcoRec"], default="EcoRec"
    )
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    sp_dir = "datasets/dlrm_pt/2022/splits"
    iter_num = 1
    batch_size = args.batch_size
    embedding_dim = 32
    tt_model = args.model
    tt_ranks = [32, 32]
    grads = torch.randn(batch_size, embedding_dim, device=device) * 1e-6

    keys = [  # largest 28 keys
        "f126",
        "f732",
        "f775",
        "f126",
        "f226",
        "f129",
        "f728",
        "f61",
        "f113",
        "f709",
        "f726",
        "f727",
        "f56",
        "f715",
        "f740",
        "f114",
        "f753",
        "f112",
        "f754",
        "f451",
        "f745",
        "f237",
        "f83",
        "f158",
        "f4",
        "f204",
        "f35",
        "f155",
    ]

    datasets = keyed_dlrm_datasets_load(
        batch_size=batch_size,
        keys=keys,
        sp_dir=sp_dir,
        iter_num=1,
        device=device,
    )
    key_num_embeddings = datasets.num_indices

    keyed_embs = {
        key: (
            create_embedding_bag(
                tt_model,
                key_num_embeddings[key],
                embedding_dim,
                tt_ranks,
                0.001,
                device,
            )
            if key_num_embeddings[key] != 0
            else None
        )
        for key in keys
    }

    iters = 5

    for i in range(iters):
        for kjt in datasets:
            for key, emb in keyed_embs.items():
                if emb is not None:
                    inds = kjt[key].values().to(device)
                    ofs = kjt[key].offsets().to(device)
                    output = emb(inds, ofs)

                    output.backward(grads)

    fwd_timers = []
    with DurationTimer() as timer:
        for i in range(iters):
            for kjt in datasets:
                for j, (key, emb) in enumerate(keyed_embs.items()):
                    if emb is not None:
                        with DurationTimer(is_sync=False) as fwd_timer:
                            cur_inds = kjt[key].values().to(device)
                            cur_ofs = kjt[key].offsets().to(device)

                            output = emb(cur_inds, cur_ofs)

                        output.backward(grads)
                        fwd_timers.append(fwd_timer)

    fwd_time = sum(fwd_timer.sync().get_duration() for fwd_timer in fwd_timers) / iters

    total_time = timer.get_duration() / iters
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3

    print("{}: FWD time: {:.3f} ms, Total time: {:.3f} ms".format(tt_model, fwd_time, total_time))
