import math
import os
import warnings
from types import ModuleType
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.cpp_extension import load

try:
    import ecorec._C as _ecorec_kernel  # type: ignore
except ImportError:
    raise
    parent_dir = os.path.dirname(os.path.abspath(__file__))

    _ecorec_kernel: ModuleType = cast(
        ModuleType,
        load(
            name="ecorec_kernel",
            sources=[
                parent_dir + "/tt_emb_kernel_wrap.cpp",
                parent_dir + "/tt_emb_cuda.cu",
            ],
            extra_cflags=["-O2"],
            extra_cuda_cflags=["-O2"],
            verbose=False,
        ),
    )


def suggested_tt_shapes(  # noqa C901
    n: int, d: int = 3, allow_round_up: bool = True
) -> List[int]:
    from itertools import cycle, islice

    # pyre-fixme[21]
    from scipy.stats import entropy
    from sympy.ntheory import factorint
    from sympy.utilities.iterables import multiset_partitions

    def _auto_shape(n: int, d: int = 3) -> List[int]:
        def _to_list(x: Dict[int, int]) -> List[int]:
            res = []
            for k, v in x.items():
                res += [k] * v
            return res

        p = _to_list(factorint(n))
        if len(p) < d:
            p = p + [1] * (d - len(p))

        def _roundrobin(*iterables):
            pending = len(iterables)
            nexts = cycle(iter(it).__next__ for it in iterables)
            while pending:
                try:
                    for next in nexts:
                        yield next()
                except StopIteration:
                    pending -= 1
                    nexts = cycle(islice(nexts, pending))

        def prepr(x: List[int]) -> Tuple:
            x = sorted(np.prod(_) for _ in x)  # type: ignore
            N = len(x)
            xf, xl = x[: N // 2], x[N // 2 :]
            return tuple(_roundrobin(xf, xl))

        raw_factors = multiset_partitions(p, d)
        clean_factors = [prepr(f) for f in raw_factors]  # type: ignore
        factors = list(set(clean_factors))
        # pyre-fixme[16]
        weights = [entropy(f) for f in factors]
        i = np.argmax(weights)
        return list(factors[i])

    def _roundup(n: int, k: int) -> int:
        return int(np.ceil(n / 10**k)) * 10**k

    if allow_round_up:
        weights = []
        for i in range(len(str(n))):
            n_i = _roundup(n, i)
            # pyre-fixme[16]
            weights.append(entropy(_auto_shape(n_i, d=d)))
        i = np.argmax(weights)
        factors = _auto_shape(_roundup(n, i), d=d)  # type: ignore
    else:
        factors = _auto_shape(n, d=d)
    return factors


def enhance_unique(
    x: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    x_sorted, back_map = torch.sort(x)

    uniques, sorted_inverse, counts = torch.unique_consecutive(
        x_sorted, return_inverse=True, return_counts=True
    )
    inverse = torch.empty_like(back_map)
    inverse[back_map] = sorted_inverse

    return uniques, inverse, counts, back_map


def _enhance_unique_with_optional_results(
    indices: Tensor,
    uniques: Optional[Tensor],
    inverse: Optional[Tensor],
    unique_counts: Optional[Tensor],
    back_map: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if uniques is None or inverse is None or unique_counts is None or back_map is None:
        uniques, inverse, unique_counts, back_map = enhance_unique(indices)

    return uniques, inverse, unique_counts, back_map


class TTLookupfunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        num_embeddings,
        embedding_dim,
        batch_size,
        learning_rate,
        rowidx,
        unique_values,
        unique_keys,
        counts,
        back_map,
        bce_idx,
        bce_map,
        bce_front_num,
        tt_p_shapes,
        tt_q_shapes,
        tt_ranks,
        tensor_p_shapes,
        tensor_q_shapes,
        tensor_tt_ranks,
        *tt_cores,
    ):
        ctx.num_embeddings = num_embeddings
        ctx.embedding_dim = embedding_dim
        ctx.batch_size = batch_size
        ctx.bce_front_num = bce_front_num
        ctx.learning_rate = learning_rate

        ctx.tt_p_shapes = tt_p_shapes
        ctx.tt_q_shapes = tt_q_shapes
        ctx.tt_ranks = tt_ranks

        ctx.save_for_backward(
            rowidx,
            unique_values,
            unique_keys,
            counts,
            back_map,
            bce_idx,
            bce_map,
            tensor_p_shapes,
            tensor_q_shapes,
            tensor_tt_ranks,
            *tt_cores,
        )

        output = _ecorec_kernel.tt_3d_forward_bag(
            num_embeddings,
            embedding_dim,
            batch_size,
            rowidx,
            unique_values,
            unique_keys,
            counts,
            back_map,
            bce_idx,
            bce_map,
            bce_front_num,
            tt_p_shapes,
            tt_q_shapes,
            tt_ranks,
            tensor_p_shapes,
            tensor_q_shapes,
            tensor_tt_ranks,
            list(tt_cores),
        )

        return output

    @staticmethod
    def backward(ctx, d_reduced: Tensor):
        num_embeddings: int = ctx.num_embeddings
        embedding_dim: int = ctx.embedding_dim
        learning_rate: float = ctx.learning_rate
        bce_front_num: int = ctx.bce_front_num

        tt_p_shapes: List[int] = ctx.tt_p_shapes
        tt_q_shapes: List[int] = ctx.tt_q_shapes
        tt_ranks: List[int] = ctx.tt_ranks

        (
            rowidx,
            unique_values,
            unique_keys,
            counts,
            back_map,
            bce_idx,
            bce_map,
            tensor_p_shapes,
            tensor_q_shapes,
            tensor_tt_ranks,
            *tt_cores,
        ) = ctx.saved_tensors

        # uni_d1 = _ecorec_kernel.aggregate_gradients(
        #     rowidx.numel(),
        #     unique_values.numel(),
        #     tt_q_shapes[0] * tt_q_shapes[1] * tt_q_shapes[2],
        #     embedding_dim,
        #     rowidx,
        #     unique_keys,
        #     d_reduced,
        # )
        # uni_d2 = _ecorec_kernel.aggregate_gradients_v1(
        #     rowidx.numel(),
        #     ctx.batch_size,
        #     embedding_dim,
        #     d_reduced,
        #     rowidx,
        #     unique_values,
        #     unique_keys,
        # )

        _ecorec_kernel.fused_tt_3d_backward_bag(
            num_embeddings,
            embedding_dim,
            learning_rate,
            rowidx,
            unique_values,
            unique_keys,
            bce_idx,
            bce_map,
            bce_front_num,
            tt_p_shapes,
            tt_q_shapes,
            tt_ranks,
            tensor_p_shapes,
            tensor_q_shapes,
            tensor_tt_ranks,
            list(tt_cores),
            d_reduced,
        )

        return (None,) * 21


class TTLookupfunctionWithUidxBatch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        num_embeddings,
        embedding_dim,
        batch_size,
        learning_rate,
        rowidx,
        unique_values,
        micro_batches,
        tt_p_shapes,
        tt_q_shapes,
        tt_ranks,
        tensor_p_shapes,
        tensor_q_shapes,
        tensor_tt_ranks,
        *tt_cores,
    ):
        ctx.num_embeddings = num_embeddings
        ctx.embedding_dim = embedding_dim
        ctx.batch_size = batch_size
        ctx.learning_rate = learning_rate

        ctx.tt_p_shapes = tt_p_shapes
        ctx.tt_q_shapes = tt_q_shapes
        ctx.tt_ranks = tt_ranks
        ctx.micro_batches = micro_batches

        ctx.save_for_backward(
            rowidx,
            unique_values,
            tensor_p_shapes,
            tensor_q_shapes,
            tensor_tt_ranks,
            *tt_cores,
        )

        output, micro_infos = _ecorec_kernel.tt_3d_forward_bag_with_uidx_batched(
            num_embeddings,
            embedding_dim,
            batch_size,
            rowidx,
            unique_values,
            tt_p_shapes,
            tt_q_shapes,
            tt_ranks,
            tensor_p_shapes,
            tensor_q_shapes,
            tensor_tt_ranks,
            micro_batches,
            list(tt_cores),
        )

        ctx.micro_infos = micro_infos

        return output

    @staticmethod
    def backward(ctx, d_reduced: Tensor):

        num_embeddings: int = ctx.num_embeddings
        embedding_dim: int = ctx.embedding_dim
        learning_rate: float = ctx.learning_rate

        tt_p_shapes: List[int] = ctx.tt_p_shapes
        tt_q_shapes: List[int] = ctx.tt_q_shapes
        tt_ranks: List[int] = ctx.tt_ranks
        micro_batches = ctx.micro_batches
        micro_infos = ctx.micro_infos

        (
            rowidx,
            unique_values,
            tensor_p_shapes,
            tensor_q_shapes,
            tensor_tt_ranks,
            *tt_cores,
        ) = ctx.saved_tensors
        
        if not d_reduced.is_contiguous():
            d_reduced = d_reduced.contiguous()
            
        optim_type = 0 # SGD

        _ecorec_kernel.fused_tt_3d_backward_bag_contraction(
            num_embeddings,
            embedding_dim,
            optim_type,
            learning_rate,
            tt_p_shapes,
            tt_q_shapes,
            tt_ranks,
            tensor_p_shapes,
            tensor_q_shapes,
            tensor_tt_ranks,
            list(tt_cores),
            d_reduced,
            micro_batches,
            micro_infos,
        )

        return (None,) * 19


def _assert_lengths_or_offsets_is_provides(
    lengths: Optional[Tensor], offsets: Optional[Tensor]
):
    assert lengths is not None or offsets is not None, (
        "Default `lengths` or `offsets` is not supported for EmbeddingBag. If you don't want to use `lengths` or `offsets`, "
        "please assign `lengths` as `torch.arange(0, indices.numel() + 1, dtype=torch.int64, device=indices.device)`"
        "or assign `offsets` as `torch.ones(indices.numel(), dtype=torch.int64, device=indices.device)`."
    )


def _compute_rowdidx_and_batch_size(
    indices: Tensor,
    lengths: Optional[Tensor],
    offsets: Optional[Tensor],
    rowidx: Optional[Tensor],
):
    _assert_lengths_or_offsets_is_provides(lengths, offsets)

    if lengths is not None:
        batch_size = lengths.numel()
        rowidx = (
            torch.repeat_interleave(
                torch.arange(batch_size, device=lengths.device, dtype=torch.int64),
                lengths,
            )
            if rowidx is None
            else rowidx
        )

    else:
        assert offsets is not None

        batch_size = offsets.numel() - 1
        rowidx = (
            _ecorec_kernel.compute_rowidx(indices, offsets)
            if rowidx is None
            else rowidx
        )
        rowidx = cast(Tensor, rowidx)

    return rowidx, batch_size


def _sharding_uniques(
    uniques: Tensor,
    unique_counts: Tensor,
    back_map: Tensor,
    rowdix: Tensor,
    num_batches: int,
) -> List[Tuple[Tensor, Tensor, Tensor]]:
    nnz = uniques.numel()
    if num_batches > nnz:
        num_batches = nnz

    quotient = nnz // num_batches
    remainder = nnz % num_batches
    num_per_batch = [quotient] * num_batches
    for i in range(remainder):
        num_per_batch[i] += 1

    offsets = torch.zeros([], device=uniques.device, dtype=torch.int64)
    start = 0
    output_list = []

    from torch.cuda import nvtx
    for i, n in enumerate(num_per_batch):
        end = start + n

        nvtx.range_push("sharding_uniques")
        micro_uniques = uniques[start:end]
        nvtx.range_pop()
        nvtx.range_push("sharding_counts")
        micro_unique_counts = unique_counts[start:end]
        micro_counts_sum = micro_unique_counts.sum()
        nvtx.range_pop()
        nvtx.range_push("sharding_back_map")
        micro_back_rowidx = torch.repeat_interleave(
            torch.arange(
                micro_unique_counts.numel(),
                device=micro_unique_counts.device,
                dtype=torch.int64,
            ),
            micro_unique_counts,
        )
        nvtx.range_pop()
        nvtx.range_push("sharding_rowidx")
        micro_back_map = back_map[offsets : offsets + micro_counts_sum]
        micro_rowidx = rowdix[micro_back_map]
        nvtx.range_pop()

        offsets += micro_counts_sum
        start = end
        output_list.append((micro_uniques, micro_back_rowidx, micro_rowidx))

    return output_list


class TTEmbeddingBag(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tt_ranks: List[int],  # [r1, r2], self.tt_ranks = [1, r1, r2, 1]
        tt_p_shapes: Optional[List[int]] = None,
        tt_q_shapes: Optional[List[int]] = None,
        optimizer: str = "SGD",
        learning_rate: float = 0.1,
        eps: float = 1.0e-10,
        sparse: bool = True,
        weight_dist: str = "normal",
        device: Optional[torch.device] = None,
        enforce_embedding_dim: bool = False,
    ) -> None:
        super(TTEmbeddingBag, self).__init__()
        assert torch.cuda.is_available()

        assert num_embeddings > 0
        assert embedding_dim > 0

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_tt_core = len(tt_ranks) + 1
        self.tt_ranks = [1] + tt_ranks + [1]
        self.sparse = sparse
        self.eps = eps

        self.tt_p_shapes = (
            suggested_tt_shapes(num_embeddings, self.num_tt_core)
            if tt_p_shapes is None
            else tt_p_shapes
        )
        self.tt_q_shapes = (
            suggested_tt_shapes(
                embedding_dim, self.num_tt_core, not enforce_embedding_dim
            )
            if tt_q_shapes is None
            else tt_q_shapes
        )

        assert len(self.tt_p_shapes) >= 2
        assert len(self.tt_p_shapes) <= 4
        assert len(tt_ranks) + 1 == len(self.tt_p_shapes)
        assert len(self.tt_p_shapes) == len(self.tt_q_shapes)
        assert len(self.tt_p_shapes) == len(self.tt_q_shapes)
        assert all(v > 0 for v in self.tt_p_shapes)
        assert all(v > 0 for v in self.tt_q_shapes)
        assert all(v > 0 for v in tt_ranks)
        assert np.prod(np.array(self.tt_p_shapes)) >= num_embeddings  # type: ignore

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_dist = weight_dist

        device = (
            device
            if device is not None
            else torch.device("cuda:{}".format(torch.cuda.current_device()))
        )

        # allocate TT cores memory
        self.tt_cores = torch.nn.ParameterList()
        for i in range(self.num_tt_core):
            self.tt_cores.append(
                torch.nn.Parameter(
                    torch.empty(
                        (
                            self.tt_p_shapes[i],
                            self.tt_ranks[i]
                            * self.tt_q_shapes[i]
                            * self.tt_ranks[i + 1],
                        ),
                        dtype=torch.float32,
                        device=device,
                    )
                )
            )

        self.reset_parameters()
        self.tensor_p_shapes = torch.tensor(self.tt_p_shapes).to(device)
        self.tensor_q_shapes = torch.tensor(self.tt_q_shapes).to(device)
        self.tensor_tt_ranks = torch.tensor(self.tt_ranks).to(device)

    def forward_gemm_compress_ratio(self):
        return self.forward_gemm_compress_ratio_t.item()

    def backward_gemm_compress_ratio(self):
        return self.backward_gemm_compress_ratio_t.item()

    def reset_parameters(self):
        if self.weight_dist == "uniform":
            lamb = 2.0 / (self.num_embeddings + self.embedding_dim)
            stddev = np.sqrt(lamb)
            tt_ranks = np.array(self.tt_ranks)
            cr_exponent = -1.0 / (2 * self.num_tt_core)
            var = np.prod(tt_ranks**cr_exponent)
            core_stddev = stddev ** (1.0 / self.num_tt_core) * var
            for i in range(self.num_tt_core):
                torch.nn.init.uniform_(self.tt_cores[i], 0.0, core_stddev)
        elif self.weight_dist == "normal":
            mu = 0.0
            sigma = 1.0 / np.sqrt(self.num_embeddings)
            scale = 1.0 / self.tt_ranks[0]
            for i in range(self.num_tt_core):
                torch.nn.init.normal_(self.tt_cores[i], mu, sigma)
                self.tt_cores[i].data *= scale
        else:
            raise ValueError("Unknown weight_dist: {}".format(self.weight_dist))

    def generate_bce_map_num(self, uniques: Tensor):
        # Maybe this should be implemented in C++/CUDA?
        bce_map = torch.arange(
            0, uniques.numel(), dtype=torch.int64, device=uniques.device
        )  # TODO: bce algorithm
        bce_front_num = bce_map.numel()  # TODO: bce algorithm
        return bce_map, bce_front_num

    def forward(
        self,
        indices: Tensor,
        offsets: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
        # NOTE: If you don't know the meanings of following arguments, please use their
        # default values and you would benefit from optimized tt-lookup algorithm.
        uniques: Optional[Tensor] = None,
        inverse: Optional[Tensor] = None,
        unique_counts: Optional[Tensor] = None,
        back_map: Optional[Tensor] = None,
        rowidx: Optional[Tensor] = None,
        bce_map: Optional[
            Tensor
        ] = None,  # Accessor to use Binary Chained Evaluation(BCE)
        bce_front_num: Optional[
            int
        ] = None,  # Number of indices joined in `front contraction`
        num_micros: int = 1,
        min_batch_count = 20000, # 0: no batch
    ):
        if min_batch_count <= 0:
            raise ValueError("min_batch_count must be greater than 0")

        from torch.cuda import nvtx

        nvtx.range_push("gen unqiques and inverse")
        (
            uniques,
            inverse,
            unique_counts,
            back_map,
        ) = _enhance_unique_with_optional_results(
            indices, uniques, inverse, unique_counts, back_map
        )
        nvtx.range_pop()

        nvtx.range_push("compute rowidx")
        rowidx, batch_size = _compute_rowdidx_and_batch_size(
            indices, lengths, offsets, rowidx
        )
        nvtx.range_pop()

        # TODO: Implement uidx-micro batch with bce algorithm
        if num_micros <= 0:
            nvtx.range_push("gen bce_map and bce_front_num")
            if bce_map is None or bce_front_num is None:
                bce_map, bce_front_num = self.generate_bce_map_num(uniques)
            bce_idx = uniques[bce_map]

            assert bce_map.numel() == uniques.numel()
            assert bce_front_num <= uniques.numel()
            nvtx.range_pop()

            nvtx.range_push("C++ forward")
            output = TTLookupfunction.apply(
                self.num_embeddings,
                self.embedding_dim,
                batch_size,
                self.learning_rate,
                rowidx,
                uniques,
                inverse,
                unique_counts,
                back_map,
                bce_idx,
                bce_map,
                bce_front_num,
                self.tt_p_shapes,
                self.tt_q_shapes,
                self.tt_ranks,
                self.tensor_p_shapes,
                self.tensor_q_shapes,
                self.tensor_tt_ranks,
                *self.tt_cores,
            )
            nvtx.range_pop()
        else:
            nnz = uniques.numel()
            batch_count = math.ceil(nnz / num_micros)
            if batch_count < min_batch_count:
                num_micros = math.ceil(nnz / min_batch_count)

            micro_batches = _sharding_uniques(
                uniques, unique_counts, back_map, rowidx, num_micros
            )

            output = TTLookupfunctionWithUidxBatch.apply(
                self.num_embeddings,
                self.embedding_dim,
                batch_size,
                self.learning_rate,
                rowidx,
                uniques,
                micro_batches,
                self.tt_p_shapes,
                self.tt_q_shapes,
                self.tt_ranks,
                self.tensor_p_shapes,
                self.tensor_q_shapes,
                self.tensor_tt_ranks,
                *self.tt_cores,
            )

        return output
