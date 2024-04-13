from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup  # type: ignore
from torchrec import KeyedTensor


def all_reduce(params, group=None, reduce_grads=False):
    for p in params:
        if isinstance(p, torch.nn.Parameter):
            if reduce_grads:
                assert p.grad is not None
                p = p.grad.data
            else:
                p = p.data
            dist.all_reduce(p, op=dist.ReduceOp.SUM, group=group)


def table_wise_embedding_all_to_all(
    group: Optional[ProcessGroup],
    keyed_embeddings: KeyedTensor,
    sharding_plan: Dict[int, List[str]],
    check: bool = False,
    return_keyed_embeddings: bool = True,
):
    group_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    global_batch_size = keyed_embeddings.values().shape[0]
    assert global_batch_size % group_size == 0
    local_batch_size = global_batch_size // group_size

    num_local_keys = len(keyed_embeddings.keys())
    assert num_local_keys == len(
        sharding_plan[rank]
    ), f"{num_local_keys}, {sharding_plan[rank]}"

    assert keyed_embeddings.values().shape[1] % num_local_keys == 0
    embedding_dim = keyed_embeddings.values().shape[1] // num_local_keys

    a2a_input = keyed_embeddings.values()  # (B x f * d)
    a2a_input_list = list(a2a_input.chunk(group_size, dim=0))
    for t in a2a_input_list:  # check no copy
        assert t.untyped_storage().data_ptr() == a2a_input.untyped_storage().data_ptr()

    a2a_output_list = [
        torch.empty(
            local_batch_size,
            len(sharding_plan[r]) * embedding_dim,
            device=keyed_embeddings.values().device,
            dtype=keyed_embeddings.values().dtype,
        )
        for r in range(group_size)
    ]

    dist.all_to_all(
        a2a_output_list, a2a_input_list, group=group
    )  # all to all buffer size: b x f * d

    global_keys = [key for r in range(group_size) for key in sharding_plan[r]]
    output_embeddings = torch.cat(a2a_output_list, dim=1)  # (b x F * d)

    if check or return_keyed_embeddings:
        output_keyed_embeddings = KeyedTensor(
            keys=global_keys,
            length_per_key=[embedding_dim for _ in global_keys],
            values=output_embeddings,
            key_dim=1,
        )

        # Check result
        if check:
            for key in sharding_plan[rank]:
                start_ofs = rank * local_batch_size
                end_ofs = (rank + 1) * local_batch_size
                assert torch.allclose(
                    keyed_embeddings[key][start_ofs:end_ofs],
                    output_keyed_embeddings[key],
                ), f"key: {key}"
            print("all to all check pass.")

        if return_keyed_embeddings:
            return output_keyed_embeddings

    return output_embeddings


def table_wise_grads_all_to_all(
    group: Optional[ProcessGroup],
    keyed_grads: KeyedTensor,
    sharding_plan: Dict[int, List[str]],
    check: bool = False,
    return_keyed_embeddings: bool = True,
):
    group_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    local_keys = sharding_plan[rank]
    global_keys = [key for r in range(group_size) for key in sharding_plan[r]]

    local_batch_size = keyed_grads.values().shape[0]
    global_batch_size = local_batch_size * group_size

    num_global_keys = len(keyed_grads.keys())
    assert num_global_keys == len(global_keys), f"{num_global_keys}, {global_keys}"

    assert keyed_grads.values().shape[1] % num_global_keys == 0
    embedding_dim = keyed_grads.values().shape[1] // num_global_keys

    length_per_rank = [0] + [
        len(sharding_plan[r]) * embedding_dim for r in range(group_size)
    ]
    cumsum_length_per_rank = np.cumsum(length_per_rank)

    a2a_input = keyed_grads.values()  # (b x F * d)
    a2a_input_list = [  # (b x f * d)
        a2a_input[
            :, cumsum_length_per_rank[r] : cumsum_length_per_rank[r + 1]
        ].contiguous()
        for r in range(group_size)
    ]
    a2a_output = torch.empty(  # (B x f * d)
        global_batch_size,
        len(local_keys) * embedding_dim,
        device=a2a_input.device,
        dtype=a2a_input.dtype,
    )
    a2a_output_list = list(a2a_output.chunk(group_size, dim=0))  # (b x f * d)

    for t in a2a_output_list:  # check no copy
        assert t.untyped_storage().data_ptr() == a2a_output.untyped_storage().data_ptr()

    dist.all_to_all(
        a2a_output_list, a2a_input_list, group=group
    )  # all to all buffer size: b x f * d

    if check or return_keyed_embeddings:
        output_keyed_embeddings = KeyedTensor(
            keys=local_keys,
            length_per_key=[embedding_dim for _ in local_keys],
            values=a2a_output,
            key_dim=1,
        )

        # Check result
        if check:
            for key in sharding_plan[rank]:
                start_ofs = rank * local_batch_size
                end_ofs = (rank + 1) * local_batch_size
                assert torch.allclose(
                    keyed_grads[key],  # (b x d)
                    output_keyed_embeddings[key][start_ofs:end_ofs, :],  # (b x d)
                ), f"key: {key} mean1: {keyed_grads[key].mean()}, mean2: {output_keyed_embeddings[key][start_ofs:end_ofs, :].mean()}"
            print("all to all check pass(grads).")

        if return_keyed_embeddings:
            return output_keyed_embeddings

    return a2a_output

