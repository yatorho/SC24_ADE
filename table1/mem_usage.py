import argparse

import torch
import torch.nn as nn

import ecorec.tt_emb.tt_emb as ecorec
import elrec_ext.Efficient_TT.efficient_tt as elrec
import ttrec.fbtembedding as fbte
from data.dlrm_ds_stats import keyed_dlrm_datasets_load
from ecorec.models import create_embedding_bag

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", "--batch_size", type=int, default=10000)
    parser.add_argument("-M", "--model", type=str, default="FBTT")
    args = parser.parse_args()

    N = int(1e8)
    M = 32
    R = [32, 32]
    d = 3
    B = args.batch_size
    model = args.model

    keys = ["f4"]
    sp_dir = "datasets/dlrm_pt/2022/splits"
    device = torch.device("cuda:0")

    datasets = keyed_dlrm_datasets_load(
        batch_size=B,
        keys=keys,
        sp_dir=sp_dir,
        iter_num=1,
        # device=device,
    )

    V = datasets[0].values().numel()
    keyed_embs = {
        key: create_embedding_bag(model, N, M, R, 0.01, device) for key in keys
    }

    print("============== {} =================".format(model))
    print("N: {}, M: {}, R: {}, B: {}, V: {}".format(N, M, R, B, V))
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
    print("param memory: {:.3f} GB".format(peak_mem))

    key = keys[0]
    kjt = datasets[0]
    jt = kjt[key]

    inds = jt.values().to(device)
    ofs = jt.offsets().to(device)
    emb = keyed_embs[key]
    output = emb(inds, ofs)

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
    print("key: {}, fwd memory: {:.4f}".format(key, peak_mem))

    output.backward(output.detach().data)

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
    print("key: {}, peak memory: {:.4f}".format(key, peak_mem))
