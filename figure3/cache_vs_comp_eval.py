import argparse

import torch
import torch.nn as nn

import ttrec.fbtembedding as fbte
from data.dlrm_ds_stats import keyed_dlrm_datasets_load
from ecorec.utils import DurationTimer


def create_embedding_bag(name, num_embeddings, embedding_dim, tt_ranks, device):
    if name == "FBTT":
        emb = fbte.FBTTEmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            tt_ranks=tt_ranks,
            enforce_embedding_dim=True,
            use_cache=False,
            cache_size=int(0.01 * num_embeddings),
            hashtbl_size=int(0.01 * num_embeddings),
        ).to(device)
    elif name == "PyTorch":
        emb = nn.EmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            mode="sum",
            sparse=True,
            include_last_offset=True,
        ).to(device)
    else:
        raise ValueError("Invalid embedding bag name: {}".format(name))
    return emb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_keys", type=int, default=30)
    parser.add_argument("--num_tt_models", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    num_keys = args.num_keys
    num_tt_models = args.num_tt_models

    keys = ["f{}".format(i) for i in range(num_keys)]
    device = torch.device(args.device)

    sp_dir = "datasets/dlrm_pt/2022/splits"
    iter_num = 1
    batch_size = 4096
    embedding_dim = 32
    tt_ranks = [32, 32]

    datasets = keyed_dlrm_datasets_load(
        batch_size=batch_size,
        keys=keys,
        sp_dir=sp_dir,
        iter_num=1,
        device=device,
    )

    models = ["FBTT", "ELRec", "EcoRec", "PyTorch"]
    tt_model = models[0]
    no_tt_model = models[3]
    keyed_models = {}

    num_tt_models_ = 0
    for key, _ in sorted(
        filter(lambda x: x[0] in keys, datasets.num_indices.items()),
        key=lambda x: x[1],
        reverse=True,
    ):
        if num_tt_models_ < num_tt_models:
            keyed_models[key] = tt_model
            num_tt_models_ += 1
        else:
            keyed_models[key] = no_tt_model

    print("num_tt_models: {}".format(num_tt_models))

    for kjt in datasets:
        for key in keys:
            kjt[key].to(device)
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
    print("dataset mem: {:.3f} GB".format(peak_mem))

    key_num_embeddings = datasets.num_indices
    keyed_embs = {
        key: (
            create_embedding_bag(
                keyed_models[key],
                key_num_embeddings[key],
                embedding_dim,
                tt_ranks,
                device,
            )
            if key_num_embeddings[key] != 0
            else None
        )
        for key in keys
    }

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
    print("static mem: {:.3f} GB".format(peak_mem))

    iters = 5
    grads = torch.randn(batch_size, embedding_dim, device=device) * 1e-6

    def _bench():
        for i in range(iters):
            for kjt in datasets:
                for key, emb in keyed_embs.items():
                    if emb is not None:
                        inds = kjt[key].values().to(device)
                        ofs = kjt[key].offsets().to(device)
                        output = emb(inds, ofs)
                        output.backward(grads)

    # warmup
    _bench()

    with DurationTimer() as timer:
        _bench()

    total_time = timer.get_duration()
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3

    print("Total time: {:.3f} ms".format(total_time))
    print("Peak memory: {:.3f} GB".format(peak_mem))
