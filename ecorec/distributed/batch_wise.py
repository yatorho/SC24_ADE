from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup  # type: ignore
from torchrec import KeyedJaggedTensor, KeyedTensor

from ..costs.sharding_plan import (KeyedCostMatrix, ShardingCost,
                                   ShardingMethod, ShardingPlan, sharding)
from ..models import MLP, TTEmbeddingLayer
from ..tt_emb import tt_emb
from ..utils import (iterate_in_micro_batches, iterate_in_num_groups, nvtx_pop,
                     nvtx_push, r0_print)
from .ops import (all_reduce, table_wise_embedding_all_to_all,
                  table_wise_grads_all_to_all)


@dataclass
class RuntimeContext:
    rank: int
    local_keys: List[str]
    global_keys: List[str]
    global_batch_size: int
    local_batch_size: int
    embedding_dim: int
    plan: ShardingPlan
    nvtx_cond: bool
    comp_stream: torch.cuda.Stream
    comm_stream: torch.cuda.Stream
    device: torch.device
    empty_embedding: Tensor  # an empty tensor with (global_batch_size x embedding_dim)

    loss_func: Callable

    num_micro_batches: int

    check_result: bool = False  # whether to check comm. OPs results

    keyed_embedding_list: Optional[List[KeyedTensor]] = None

    init_grad: Optional[Tensor] = None
    



class BatchWisePipelineFunction:
    @staticmethod
    def forward(
        fb: KeyedJaggedTensor,
        keyed_embs,
        ctx: RuntimeContext,
    ):
        device = ctx.device
        rank = ctx.rank
        local_keys = ctx.local_keys
        empty_embedding = ctx.empty_embedding
        plan = ctx.plan
        nvtx_cond = ctx.nvtx_cond
        global_batch_size = ctx.global_batch_size
        local_batch_size = ctx.local_batch_size
        embedding_dim = ctx.embedding_dim
        check_result = ctx.check_result

        comp_stream = ctx.comp_stream
        comm_stream = ctx.comm_stream

        num_micro_batches = ctx.num_micro_batches
        global_micro_batch_size = global_batch_size // num_micro_batches
        local_micro_batch_size = local_batch_size // num_micro_batches
        assert (
            local_batch_size % num_micro_batches == 0
            and global_batch_size % num_micro_batches == 0
        ), (
            "global batch size and local batch size should be divisible by num_micro_batches. "
            f"But got [{global_batch_size} / {local_batch_size}] % {num_micro_batches} != 0"
        )

        group_size = dist.get_world_size(group=None)
        global_keys = [key for r in range(group_size) for key in plan[r]]

        # for i, group_keys in enumerate(group_key_list_per_rank[rank]):
        output_embedding_list: List[Tensor] = []
        keyed_embedding_list: List[KeyedTensor] = []
        input_record = {} if check_result else None

        for i, micro_kjt in enumerate(iterate_in_micro_batches(fb, num_micro_batches)):
            start_ofs = i * global_micro_batch_size
            end_ofs = (i + 1) * global_micro_batch_size

            nvtx_push("to device", nvtx_cond)
            micro_keyed_requests = {
                key: (
                    micro_kjt[key].values().to(device),
                    micro_kjt[key].offsets().to(device),
                )
                for key in local_keys
            }
            nvtx_pop(nvtx_cond)

            with torch.cuda.stream(comp_stream):
                # For debugging, avoid to use comprehension
                embeddings_dict = {}
                for key in local_keys:
                    nvtx_push("fwd(key:{})".format(key), nvtx_cond)
                    emb = keyed_embs[key]
                    if emb is None:
                        empty_embedding_ = empty_embedding[start_ofs:end_ofs].clone()
                        empty_embedding_.requires_grad = True
                        embeddings_dict[key] = empty_embedding_
                    else:
                        indices, offsets = micro_keyed_requests[key]
                        embeddings_dict[key] = emb(indices, offsets) # (mB x f * d)
                    nvtx_pop(nvtx_cond)

                if check_result:
                    assert input_record is not None
                    for key in local_keys:
                        if key not in input_record: 
                            input_record[key] = [] 

                        input_record[key].append(embeddings_dict[key]) 

                nvtx_push("gen key tensor", nvtx_cond)
                keyed_embeddings = KeyedTensor.from_tensor_list(
                    list(embeddings_dict.keys()), list(embeddings_dict.values())
                ) # key, (mB x f * d)
                keyed_embedding_list.append(keyed_embeddings)
                nvtx_pop(nvtx_cond)

            comm_stream.wait_stream(comp_stream)
            with torch.cuda.stream(comm_stream):
                # All to All communication
                nvtx_push("AlltoAll", nvtx_cond)
                embeddings = table_wise_embedding_all_to_all(
                    None,
                    keyed_embeddings,
                    plan,
                    check=check_result,
                    return_keyed_embeddings=False,
                )
                embeddings = cast(Tensor, embeddings)
                assert embeddings.shape == (  # (mb x F * d)
                    local_micro_batch_size,
                    len(global_keys) * embedding_dim,
                ), f"{embeddings.shape}, {local_micro_batch_size}, {len(global_keys)}, {embedding_dim}"
                output_embedding_list.append(embeddings)
                nvtx_pop(nvtx_cond)

        # sync
        comp_stream.wait_stream(comm_stream)
        ctx.keyed_embedding_list = keyed_embedding_list

        output_embeddings = torch.cat(output_embedding_list, dim=0)  # (b x F * d)
        out_keyed_embeddings = KeyedTensor(
            keys=global_keys,
            length_per_key=[embedding_dim for _ in global_keys],
            values=output_embeddings,
            key_dim=1,
        )

        if check_result:
            for key in local_keys:
                start_ofs = rank * local_batch_size
                end_ofs = (rank + 1) * local_batch_size

                input_embedding = torch.cat(input_record[key], dim=0) # type: ignore
                assert torch.allclose(
                    input_embedding[start_ofs:end_ofs], out_keyed_embeddings[key]
                ), f"key: {key}, mean1: {input_embedding[start_ofs:end_ofs].mean()}, mean2: {out_keyed_embeddings[key].mean()}"

        return out_keyed_embeddings

    @staticmethod
    def backward(keyed_grads: KeyedTensor, ctx: RuntimeContext):
        global_keys = ctx.global_keys
        plan = ctx.plan
        nvtx_cond = ctx.nvtx_cond
        local_batch_size = ctx.local_batch_size
        check_result = ctx.check_result

        comp_stream = ctx.comp_stream
        comm_stream = ctx.comm_stream

        num_micro_batches = ctx.num_micro_batches
        local_micro_batch_size = local_batch_size // num_micro_batches
        
        comm_stream.wait_stream(comp_stream)
        keyed_embedding_list = ctx.keyed_embedding_list # List[key, (mB x f * d)]
        # keyed_grads: key, (b x F * d)
        assert keyed_embedding_list is not None

        for i in range(num_micro_batches):
            start_ofs = i * local_micro_batch_size
            end_ofs = (i + 1) * local_micro_batch_size
            with torch.cuda.stream(comm_stream):
                micro_keyed_grads = KeyedTensor(
                    keys=global_keys,
                    length_per_key=keyed_grads.length_per_key(),
                    values=keyed_grads.values()[start_ofs:end_ofs],
                )

                # All to All communication
                nvtx_push("AlltoAll(bwd)", nvtx_cond)
                out_micro_keyed_grads = table_wise_grads_all_to_all(
                    None, micro_keyed_grads, plan, check=check_result
                ) # key, (mB x f * d)
                out_micro_keyed_grads = cast(KeyedTensor, out_micro_keyed_grads)
                nvtx_pop(nvtx_cond)

            comp_stream.wait_stream(comm_stream)
            with torch.cuda.stream(comp_stream):
                nvtx_push("bwd", nvtx_cond)
                grads = out_micro_keyed_grads.values()
                embedding = keyed_embedding_list[i].values()
                assert embedding.requires_grad and embedding.shape == grads.shape

                embedding.backward(grads)
                nvtx_pop(nvtx_cond)




class BatchWisePipelineEngine:
    @staticmethod
    def generate_plan(
        keyed_cost_matrix: KeyedCostMatrix,
        sharding_method=ShardingMethod.GREED,
    ):
        plan, costs = sharding(keyed_cost_matrix, sharding_method)

        return plan, costs


    @staticmethod
    def execute_batch(
        batch: Tuple[KeyedJaggedTensor, Tensor, Tensor],
        embedding_layer: TTEmbeddingLayer,
        mlps: MLP,
        ctx: RuntimeContext,
    ):

        sparse_batch, dense_batch, label = batch

        embed_dense = mlps.dense_arch(dense_batch)
        embed_sparse = BatchWisePipelineFunction.forward(
            sparse_batch, embedding_layer.keyed_embs, ctx
        )  # embedding
        embed_sparse = cast(KeyedTensor, embed_sparse)  # (b x F * d)
        embed_sparse.values().requires_grad = True
        sparse_emb_values = embed_sparse.values().view(
            -1, len(ctx.global_keys), ctx.embedding_dim
        )

        embed_inter = mlps.inter_arch(
            dense_features=embed_dense,
            sparse_features=sparse_emb_values,
        )  # interaction arch
        logits = mlps.over_arch(embed_inter)  # over arch

        if ctx.init_grad is not None:
            gradient = ctx.init_grad
        else:
            assert label.shape == logits.shape
            gradient = ctx.loss_func(logits, label)

        logits.backward(gradient)

        # mlp allreduce
        all_reduce(mlps.dense_arch.parameters(), reduce_grads=True)
        all_reduce(mlps.inter_arch.parameters(), reduce_grads=True)
        all_reduce(mlps.over_arch.parameters(), reduce_grads=True)

        # backward embedding
        sparse_grad_values = embed_sparse.values().grad
        assert sparse_grad_values is not None
        sparse_keyed_grad = KeyedTensor( # (b x F * d)
            keys=embed_sparse.keys(),
            length_per_key=embed_sparse.length_per_key(),
            values=sparse_grad_values,
        )

        BatchWisePipelineFunction.backward(sparse_keyed_grad, ctx)




