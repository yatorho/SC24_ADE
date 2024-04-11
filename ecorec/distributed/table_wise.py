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
from ..utils import iterate_in_num_groups, nvtx_pop, nvtx_push, r0_print
from .ops import (all_reduce, table_wise_embedding_all_to_all,
                  table_wise_grads_all_to_all)


def reorder_table_execution_plan(
    plan: ShardingPlan, 
    keyed_matrix: KeyedCostMatrix,
    is_sorted: bool = True,
) -> ShardingPlan:
    """
    Reduce the bubble in table wise pipeline via reordering the execution plan of features.
    For a simple implementation, we just order the features by their execution time.

    Args:
        plan: the features' placement plan generatd from greedy algorithm.

    Returns:
        the reordered plan.
    """
    # # reverse plan
    if is_sorted:
        reordered_plan = {
            r: list(reversed(lst)) for r, lst in plan.items()
        }  # !!!!! reverse plan!
    else:
        reordered_plan = {
            r: sorted(lst, key=lambda x: keyed_matrix[x][r]) for r, lst in plan.items()
        }

    return reordered_plan


@dataclass
class RuntimeContext:
    rank: int
    global_keys: List[str]
    local_batch_size: int
    embedding_dim: int
    plan: ShardingPlan
    nvtx_cond: bool
    comp_stream: torch.cuda.Stream
    comm_stream: torch.cuda.Stream
    device: torch.device
    empty_embedding: Tensor  # an empty tensor with (global_batch_size x embedding_dim)

    loss_func: Callable

    num_micro_keys: int  # table-wise pipelining granularity
    num_micro_uidx: int  # C-M/M-C pattern micro-batching strategy
    skew_degree: float = 1.0  # slope feature counts distribution

    check_result: bool = False  # whether to check comm. OPs results

    group_key_list_per_rank: Optional[Dict[int, List[Sequence[str]]]] = None
    back_group_key_list_per_rank: Optional[Dict[int, List[Sequence[str]]]] = None
    embedding_dict: Optional[Dict[str, Tensor]] = None

    init_grad: Optional[Tensor] = None


class TableUidxWisePipelineFunction:
    @staticmethod
    def forward(
        fb: KeyedJaggedTensor,
        keyed_embs,
        ctx: RuntimeContext,
    ) -> KeyedTensor:
        device = ctx.device
        rank = ctx.rank
        empty_embedding = ctx.empty_embedding
        plan = ctx.plan
        nvtx_cond = ctx.nvtx_cond
        local_batch_size = ctx.local_batch_size
        check_result = ctx.check_result

        comp_stream = ctx.comp_stream
        comm_stream = ctx.comm_stream

        num_micro_keys = ctx.num_micro_keys
        num_micro_uidx = ctx.num_micro_uidx
        skew_degree = ctx.skew_degree

        # if "group_key_list_per_rank" not in ctx:
        if ctx.group_key_list_per_rank is None:
            group_key_list_per_rank = {
                r: list(
                    iterate_in_num_groups(
                        plan[r], num_micro_keys, skew_degree, decreasing=True
                    )
                )
                for r in plan.keys()
            }
            ctx.group_key_list_per_rank = group_key_list_per_rank
            r0_print(
                f"group_key_len_per_rank: { {r: [len(lst) for lst in group_key_list_per_rank[r]] for r in group_key_list_per_rank.keys()}}"
            )
        else:
            group_key_list_per_rank = ctx.group_key_list_per_rank

        group_key_list_per_rank = cast(
            Dict[int, List[List[str]]], group_key_list_per_rank
        )

        embedding_dict: Dict[str, Tensor] = {}
        out_keyed_embedding_list: List[KeyedTensor] = []
        for i, group_keys in enumerate(group_key_list_per_rank[rank]):
            group_plan = {r: group_key_list_per_rank[r][i] for r in plan.keys()}

            with torch.cuda.stream(comp_stream):
                nvtx_push("to device", nvtx_cond)
                keyed_requests = {
                    key: (fb[key].values().to(device), fb[key].offsets().to(device))
                    for key in group_keys
                }
                nvtx_pop(nvtx_cond)

                # For debugging, avoid to use comprehension
                keyed_embeddings = {}
                for key in group_keys:
                    nvtx_push("fwd(key:{})".format(key), nvtx_cond)
                    emb = keyed_embs[key]
                    if emb is None:
                        empty_embedding_ = empty_embedding.clone()
                        empty_embedding_.requires_grad = True
                        keyed_embeddings[key] = empty_embedding_
                    else:
                        if isinstance(emb, tt_emb.TTEmbeddingBag):
                            keyed_embeddings[key] = emb(
                                *keyed_requests[key],
                                min_batch_count=20000,
                                num_micros=num_micro_uidx,
                            )
                        else:
                            keyed_embeddings[key] = emb(*keyed_requests[key])
                    nvtx_pop(nvtx_cond)

                embedding_dict.update(keyed_embeddings)

                nvtx_push("gen key tensor", nvtx_cond)
                keyed_embeddings = KeyedTensor.from_tensor_list(
                    list(keyed_embeddings.keys()), list(keyed_embeddings.values())
                )
                nvtx_pop(nvtx_cond)

            comm_stream.wait_stream(comp_stream)
            with torch.cuda.stream(comm_stream):
                # All to All communication
                nvtx_push("AlltoAll", nvtx_cond)
                group_keyed_embeddings = table_wise_embedding_all_to_all(
                    None, keyed_embeddings, group_plan, check=check_result
                )
                out_keyed_embedding_list.append(group_keyed_embeddings)  # type: ignore
                nvtx_pop(nvtx_cond)

        # sync
        comp_stream.wait_stream(comm_stream)
        ctx.embedding_dict = embedding_dict

        regroup_keys = [[k] for kt in out_keyed_embedding_list for k in kt.keys()]
        out_keys = [k for kl in regroup_keys for k in kl]
        regroup_dict = KeyedTensor.regroup_as_dict(
            out_keyed_embedding_list, regroup_keys, out_keys
        )

        out_keyed_embeddings = KeyedTensor.from_tensor_list(
            list(regroup_dict.keys()), list(regroup_dict.values())
        )

        if check_result:
            for key in plan[rank]:  # Check result
                start_ofs = rank * local_batch_size
                end_ofs = (rank + 1) * local_batch_size
                assert torch.allclose(
                    embedding_dict[key][start_ofs:end_ofs], out_keyed_embeddings[key]
                ), f"key: {key}"  # type: ignore

        return out_keyed_embeddings

    @staticmethod
    def backward(keyed_grads: KeyedTensor, ctx: RuntimeContext):
        rank = ctx.rank
        plan = ctx.plan
        nvtx_cond = ctx.nvtx_cond
        check_result = ctx.check_result

        world_size = dist.get_world_size(group=None)

        comp_stream = ctx.comp_stream
        comm_stream = ctx.comm_stream

        num_micro_keys = ctx.num_micro_keys
        skew_degree = ctx.skew_degree

        if ctx.back_group_key_list_per_rank is None:
            back_group_key_list_per_rank = {
                r: list(
                    iterate_in_num_groups(
                        list(reversed(plan[r])),
                        num_micro_keys,
                        skew_degree,
                        decreasing=False,
                    )
                )
                for r in plan.keys()
            }
            ctx.back_group_key_list_per_rank = back_group_key_list_per_rank
            r0_print(
                f"back_group_key_len_per_rank: { {r: [len(lst) for lst in back_group_key_list_per_rank[r]] for r in back_group_key_list_per_rank.keys()}}"
            )
        else:
            back_group_key_list_per_rank = ctx.back_group_key_list_per_rank

        back_group_key_list_per_rank = cast(
            Dict[int, List[List[str]]], back_group_key_list_per_rank
        )

        # sync
        comm_stream.wait_stream(comp_stream)
        embedding_dict = ctx.embedding_dict
        assert embedding_dict is not None

        for i, group_keys in enumerate(back_group_key_list_per_rank[rank]):
            group_plan = {r: back_group_key_list_per_rank[r][i] for r in plan.keys()}
            with torch.cuda.stream(comm_stream):
                global_group_keys = [
                    key for r in range(world_size) for key in group_plan[r]
                ]
                group_keyed_grads = KeyedTensor.from_tensor_list(  # (b x mF * d)
                    keys=global_group_keys,
                    tensors=[keyed_grads[key] for key in global_group_keys],
                )

                # All to All communication
                nvtx_push("AlltoAll(bwd)", nvtx_cond)
                out_group_keyed_grads = table_wise_grads_all_to_all(
                    None, group_keyed_grads, group_plan, check=check_result
                )
                out_group_keyed_grads = cast(KeyedTensor, out_group_keyed_grads)
                nvtx_pop(nvtx_cond)

            comp_stream.wait_stream(comm_stream)
            with torch.cuda.stream(comp_stream):
                # For debugging, avoid to use comprehension
                for key in group_keys:
                    nvtx_push("bwd(key:{})".format(key), nvtx_cond)
                    grads = out_group_keyed_grads[key]
                    embedding = embedding_dict[key]
                    assert embedding.requires_grad and embedding.shape == grads.shape

                    embedding.backward(grads)
                    nvtx_pop(nvtx_cond)


class TableWisePipelineEngine:
    @staticmethod
    def generate_plan(
        keyed_cost_matrix: KeyedCostMatrix,
        reordering_features: bool = True,
        sharding_method=ShardingMethod.GREED,
    ):
        plan, costs = sharding(keyed_cost_matrix, sharding_method)

        if reordering_features:
            plan = reorder_table_execution_plan(plan, keyed_cost_matrix)

        return plan, costs


    @staticmethod
    def execute_batch(
        batch: Tuple[KeyedJaggedTensor, Tensor, Tensor],
        embedding_layer: TTEmbeddingLayer,
        mlps: MLP,
        runtime_ctx: RuntimeContext,
    ):
        sparse_batch, dense_batch, label = batch

        embed_dense = mlps.dense_arch(dense_batch)
        embed_sparse = TableUidxWisePipelineFunction.forward(
            sparse_batch, embedding_layer.keyed_embs, runtime_ctx
        )  # embedding

        embed_sparse = cast(KeyedTensor, embed_sparse)  # (b x F * d)
        embed_sparse.values().requires_grad = True
        sparse_emb_values = embed_sparse.values().view(
            -1, len(runtime_ctx.global_keys), runtime_ctx.embedding_dim
        )

        embed_inter = mlps.inter_arch(
            dense_features=embed_dense,
            sparse_features=sparse_emb_values,
        )  # interaction arch
        logits = mlps.over_arch(embed_inter)  # over arch

        if runtime_ctx.init_grad is not None:
            gradient = runtime_ctx.init_grad
        else:
            assert label.shape == logits.shape
            gradient = runtime_ctx.loss_func(logits, label)

        logits.backward(gradient)

        # mlp allreduce
        all_reduce(mlps.dense_arch.parameters(), reduce_grads=True)
        all_reduce(mlps.inter_arch.parameters(), reduce_grads=True)
        all_reduce(mlps.over_arch.parameters(), reduce_grads=True)

        # backward embedding
        sparse_grad_values = embed_sparse.values().grad
        assert sparse_grad_values is not None
        sparse_keyed_grad = KeyedTensor(
            keys=embed_sparse.keys(),
            length_per_key=embed_sparse.length_per_key(),
            values=sparse_grad_values,
        )

        TableUidxWisePipelineFunction.backward(sparse_keyed_grad, runtime_ctx)


