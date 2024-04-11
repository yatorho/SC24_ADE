from typing import List

import numpy as np
import torch
from torchrec import JaggedTensor

import ecorec.tt_emb.tt_emb as ecorec
from data.dlrm_ds_stats import keyed_dlrm_datasets_load


def tt_inds_gen(indices, tt_p_shapes) -> List[torch.Tensor]:
    tt_strides = [np.prod(tt_p_shapes[i:]) for i in range(1, len(tt_p_shapes))] + [1]

    tt_x_idx = []
    offset = 0
    for i in range(len(tt_strides)):
        v = torch.div(indices - offset, tt_strides[i], rounding_mode="trunc")
        tt_x_idx.append(v)
        offset += v * tt_strides[i]

    return tt_x_idx


def reuse_matrix(tt_idxa, tt_idxb, lena=None, lenb=None):
    assert tt_idxa.numel() == tt_idxb.numel()
    lena = tt_idxa.max().item() + 1 if lena is None else lena
    lenb = tt_idxb.max().item() + 1 if lenb is None else lenb
    matrix = torch.bincount(tt_idxa * lenb + tt_idxb, minlength=lena * lenb).reshape(
        lena, lenb
    )
    return matrix


def reuse_nd_matrix(tt_idxs, tt_p_shapes, nd):
    if nd > len(tt_p_shapes):
        raise ValueError("nd must be less than or equal to len(tt_p_shapes)")

    tt_p_shapes = tt_p_shapes[:nd]
    tt_strides = [np.prod(tt_p_shapes[i:]) for i in range(1, len(tt_p_shapes))] + [1]

    tt_ofss = torch.zeros_like(tt_idxs[0])
    for i in range(nd):
        tt_ofss += tt_idxs[i] * tt_strides[i]

    nd_matrix = torch.bincount(tt_ofss, minlength=np.prod(tt_p_shapes)).reshape(
        tt_p_shapes
    )
    return nd_matrix


def unqiue2_pair_cnt(jt: JaggedTensor, shape, front=True):
    values = jt.values()
    uniques, inverse, counts, back_map = ecorec.enhance_unique(values)
    tt_idx = tt_inds_gen(uniques, shape)

    gemm_cnt = 0
    if front:
        rmat = reuse_matrix(tt_idx[0], tt_idx[1], shape[0], shape[1])
        gemm_cnt = (rmat != 0).sum().item()
    else:
        rmat = reuse_matrix(tt_idx[1], tt_idx[2], shape[1], shape[2])
        gemm_cnt = (rmat != 0).sum().item()

    return gemm_cnt


def unique3_pair_cnt(jt: JaggedTensor, shape):
    values = jt.values()
    tt_idx = tt_inds_gen(values, shape)

    gemm_cnt = (reuse_nd_matrix(tt_idx, shape, 3) != 0).sum().item()

    return gemm_cnt


def unique_idx_cnt(jt: JaggedTensor):
    values = jt.values()
    uniques, inverse, counts, back_map = ecorec.enhance_unique(values)

    return uniques.numel()


def micro_Uidx_gemm_cnt(jt: JaggedTensor, shape, num_micro, front=True):
    values = jt.values()
    uniques, inverse, counts, back_map = ecorec.enhance_unique(values)

    tt_idx = tt_inds_gen(uniques, shape)

    numel = tt_idx[0].numel()
    div = numel // num_micro
    rem = numel % num_micro
    num_per_micro = [div] * num_micro
    for i in range(rem):
        num_per_micro[i] += 1

    gemm_cnt = 0
    cumsum = 0
    if front:
        for len in num_per_micro:
            start_ofs = cumsum
            end_ofs = start_ofs + len
            cumsum += len

            rmat = reuse_matrix(
                tt_idx[0][start_ofs:end_ofs],
                tt_idx[1][start_ofs:end_ofs],
                shape[0],
                shape[1],
            )
            gemm_cnt += (rmat != 0).sum().item()
    else:
        for len in num_per_micro:
            start_ofs = cumsum
            end_ofs = start_ofs + len
            cumsum += len

            rmat = reuse_matrix(
                tt_idx[1][start_ofs:end_ofs],
                tt_idx[2][start_ofs:end_ofs],
                shape[1],
                shape[2],
            )
            gemm_cnt += (rmat != 0).sum().item()

    gemm_cnt += uniques.numel()

    return gemm_cnt


if __name__ == '__main__':
    local_batch_sizes = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
    global_keys = ["f0", "f4", "f44", "f94", "f125"]
    sp_dir = "datasets/dlrm_pt/2022/splits"
    device = torch.device("cuda:0")

    for key in global_keys:
        for local_batch_size in local_batch_sizes:
            feature_batches = keyed_dlrm_datasets_load(
                batch_size=local_batch_size,
                keys=global_keys,
                sp_dir=sp_dir,
                iter_num=1,
                device=device
            )
            fb = feature_batches[0]
            tt_p_shapes = ecorec.suggested_tt_shapes(feature_batches.num_indices[key], 4)
            values, offsets, lengths = (
                fb[key].values(),
                fb[key].offsets(),
                fb[key].lengths(),
            )

            if values.numel() == 0:
                continue

            indices = values.numel()
            unique = unique_idx_cnt(fb[key]) / indices
            ttidx_2pair = unqiue2_pair_cnt(fb[key], tt_p_shapes) / indices
            ttidx_3pair = unique3_pair_cnt(fb[key], tt_p_shapes) / indices

            indices = 1

            print(
                "key: {}, batch_size: {}, #indices: {}, #unique: {:.4f}, #ttidx_2pair: {:.4f}, #ttidx_3pair: {:.4f}".format(
                    key, local_batch_size, indices, unique, ttidx_2pair, ttidx_3pair
                )
            )
