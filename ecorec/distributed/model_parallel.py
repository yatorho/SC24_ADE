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


@dataclass
class RuntimeContext:
    local_keys: List[str]
    global_keys: List[str]
    embedding_dim: int
    plan: ShardingPlan
    nvtx_cond: bool
    device: torch.device
    empty_embedding: Tensor  # an empty tensor with (global_batch_size x embedding_dim)

    loss_func: Callable

    check_result: bool = False  # whether to check comm. OPs results

    init_grad: Optional[Tensor] = None
    keyed_embeddings: Optional[KeyedTensor] = None


class ModelParallelFunction:
    @staticmethod
    def forward(
        fb: KeyedJaggedTensor,
        keyed_embs,
        ctx: RuntimeContext,
    ):
        device = ctx.device
        local_keys = ctx.local_keys
        empty_embedding = ctx.empty_embedding
        plan = ctx.plan
        nvtx_cond = ctx.nvtx_cond
        check_result = ctx.check_result

        nvtx_push("to device", nvtx_cond)
        keyed_requests = {
            key: (fb[key].values().to(device), fb[key].offsets().to(device))
            for key in local_keys
        }
        nvtx_pop(nvtx_cond)

        # For debugging, avoid to use comprehension
        keyed_embeddings = {}
        for key in local_keys:
            nvtx_push("fwd(key:{})".format(key), nvtx_cond)
            emb = keyed_embs[key]
            if emb is None:
                empty_embedding_ = empty_embedding.clone()
                empty_embedding_.requires_grad = True
                keyed_embeddings[key] = empty_embedding_
            else:
                keyed_embeddings[key] = emb(*keyed_requests[key])
            nvtx_pop(nvtx_cond)

        nvtx_push("gen key tensor", nvtx_cond)
        keyed_embeddings = KeyedTensor.from_tensor_list(
            list(keyed_embeddings.keys()), list(keyed_embeddings.values())
        )
        nvtx_pop(nvtx_cond)

        assert keyed_embeddings.values().requires_grad
        ctx.keyed_embeddings = keyed_embeddings

        # All to All communication
        nvtx_push("AlltoAll", nvtx_cond)
        out_keyed_embeddings = table_wise_embedding_all_to_all(
            None, keyed_embeddings, plan, check=check_result
        )
        nvtx_pop(nvtx_cond)

        return out_keyed_embeddings
    
    @staticmethod
    def backward(keyed_grads: KeyedTensor, ctx: RuntimeContext):
        device = ctx.device
        local_keys = ctx.local_keys
        empty_embedding = ctx.empty_embedding
        plan = ctx.plan
        nvtx_cond = ctx.nvtx_cond
        check_result = ctx.check_result

        nvtx_push("AlltoAll", nvtx_cond)
        out_keyed_grads = table_wise_grads_all_to_all(
            None, keyed_grads, plan, check=check_result
        ).values()
        nvtx_pop(nvtx_cond)


        keyed_embeddings = ctx.keyed_embeddings
        assert keyed_embeddings is not None
        keyed_embeddings = keyed_embeddings.values()  # (B x f * d)
        assert keyed_embeddings.requires_grad
        assert keyed_embeddings.shape == out_keyed_grads.shape

        assert out_keyed_grads.is_contiguous()

        nvtx_push("backward", nvtx_cond)
        keyed_embeddings.backward(out_keyed_grads)
        nvtx_pop(nvtx_cond)


class ModelParallelEngine:
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
        embed_sparse = ModelParallelFunction.forward(
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
        sparse_keyed_grad = KeyedTensor(
            keys=embed_sparse.keys(),
            length_per_key=embed_sparse.length_per_key(),
            values=sparse_grad_values,
        )

        ModelParallelFunction.backward(sparse_keyed_grad, ctx)
