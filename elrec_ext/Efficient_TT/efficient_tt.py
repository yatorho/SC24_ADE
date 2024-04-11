import os
from types import ModuleType
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from torch import nn
from torch.utils.cpp_extension import load

try:
    import elrec_kernel as _elrec_kernel  # type: ignore
except ImportError:
    parent_dir = os.path.dirname(os.path.abspath(__file__))

    _elrec_kernel: ModuleType = cast(
        ModuleType,
        load(
            name="elrec_kernel",
            sources=[
                parent_dir + "/efficient_kernel_wrap.cpp", 
                parent_dir + "/efficient_tt_cuda.cu", 
            ],
            extra_cflags=["-O2"],
            extra_cuda_cflags=["-O2"],
            verbose=False,
        ),
    )

Eff_TT_embedding_cuda = _elrec_kernel

# parent_dir = os.path.dirname(os.path.abspath(__file__))
# Eff_TT_embedding_cuda = load(name="efficient_tt_table", sources=[
#     parent_dir + "/efficient_kernel_wrap.cpp", 
#     parent_dir + "/efficient_tt_cuda.cu", 
#     ], verbose=False)


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
            x = sorted(np.prod(_) for _ in x)
            N = len(x)
            xf, xl = x[: N // 2], x[N // 2 :]
            return tuple(_roundrobin(xf, xl))

        raw_factors = multiset_partitions(p, d)
        clean_factors = [prepr(f) for f in raw_factors]
        factors = list(set(clean_factors))
        # pyre-fixme[16]
        weights = [entropy(f) for f in factors]
        i = np.argmax(weights)
        return list(factors[i])

    def _roundup(n: int, k: int) -> int:
        return int(np.ceil(n / 10 ** k)) * 10 ** k

    if allow_round_up:
        weights = []
        for i in range(len(str(n))):
            n_i = _roundup(n, i)
            # pyre-fixme[16]
            weights.append(entropy(_auto_shape(n_i, d=d)))
        i = np.argmax(weights)
        factors = _auto_shape(_roundup(n, i), d=d)
    else:
        factors = _auto_shape(n, d=d)
    return factors


class TT_core_function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        learning_rate,
        batch_size,
        table_length,
        feature_dim,
        indices,
        tt_p_shapes, 
        tt_q_shapes, 
        tt_ranks, 
        tensor_p_shape, 
        tensor_q_shape, 
        tensor_tt_ranks,
        sorted_idx,
        sorted_key,
        tensor_group_map,
        tensor_group_flag,
        tensor_group_idx,
        tensor_cache,
        tensor_batch_cnt,
        *tt_cores,
    ):
        ctx.learning_rate = learning_rate
        ctx.tt_p_shapes = tt_p_shapes
        ctx.tt_q_shapes = tt_q_shapes
        ctx.tt_ranks = tt_ranks
        ctx.tensor_p_shape = tensor_p_shape
        ctx.tensor_q_shape = tensor_q_shape
        ctx.tensor_tt_ranks = tensor_tt_ranks
        ctx.table_length = table_length
        ctx.feature_dim = feature_dim
        ctx.batch_size = batch_size
        ctx.tt_cores = tt_cores
        ctx.sorted_idx = sorted_idx
        ctx.sorted_key = sorted_key

        ctx.save_for_backward(
            indices,
        )

        # breakpoint()
        output = Eff_TT_embedding_cuda.Eff_TT_forward(
            batch_size,
            table_length, # need
            feature_dim, # need
            indices,
            tt_p_shapes,
            tt_q_shapes,
            tt_ranks,
            tensor_p_shape,
            tensor_q_shape,
            tensor_tt_ranks,
            list(ctx.tt_cores),
            tensor_group_map,
            tensor_group_flag,
            tensor_group_idx,
            tensor_cache,
            tensor_batch_cnt
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor)-> Tuple[torch.Tensor]:
        
        
        indices = ctx.saved_tensors

        if ctx.sorted_key == None:
            sorted_idx, sorted_key = indices[0].unique(sorted=True, return_inverse=True)
        # Eff_TT_embedding_cuda.Eff_TT_backward(
        # Eff_TT_embedding_cuda.Fused_Eff_TT_backward(
            Eff_TT_embedding_cuda.Fused_Extra_Eff_TT_backward(
                ctx.batch_size,
                ctx.table_length,
                ctx.feature_dim,
                ctx.learning_rate,

                indices[0],
                ctx.tt_p_shapes, 
                ctx.tt_q_shapes,
                ctx.tt_ranks, 
                ctx.tensor_p_shape,
                ctx.tensor_q_shape,
                ctx.tensor_tt_ranks,
                grad_output,
                list(ctx.tt_cores),
                sorted_idx,
                sorted_key,
            )
        else:
            Eff_TT_embedding_cuda.Fused_Extra_Eff_TT_backward(
                ctx.batch_size,
                ctx.table_length,
                ctx.feature_dim,
                ctx.learning_rate,

                indices[0],
                ctx.tt_p_shapes, 
                ctx.tt_q_shapes,
                ctx.tt_ranks, 
                ctx.tensor_p_shape,
                ctx.tensor_q_shape,
                ctx.tensor_tt_ranks,
                grad_output,
                list(ctx.tt_cores),
                ctx.sorted_idx,
                ctx.sorted_key,
            )


        # Eff_TT_embedding_cuda.Eff_TT_backward(
        # # Eff_TT_embedding_cuda.Fused_Eff_TT_backward(
        #     ctx.batch_size,
        #     ctx.table_length,
        #     ctx.feature_dim,
        #     0.1,

        #     indices[0],
        #     ctx.tt_p_shapes, 
        #     ctx.tt_q_shapes,
        #     ctx.tt_ranks, 
        #     ctx.tensor_p_shape,
        #     ctx.tensor_q_shape,
        #     ctx.tensor_tt_ranks,
        #     grad_output,
        #     list(ctx.tt_cores)
        # )

        return tuple(
            [
                None,  
                None,  
                None,  
                None,  
                None,  
                None,  
                None,  
                None,  
                None,  
                None,  
                None,  
                None,  
                None,  
                None,  
                None,  
                None,
                None,
                None, # tt_core0
                None, # tt_core1
                None, # tt_core2
            ]
        )


class TT_core_bag_function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        learning_rate,
        batch_size,  # real batch cnt
        table_length,
        feature_dim,
        indices,
        offsets,
        rowidx,
        tt_p_shapes, 
        tt_q_shapes, 
        tt_ranks, 
        tensor_p_shape, 
        tensor_q_shape, 
        tensor_tt_ranks,
        sorted_idx,
        sorted_key,
        tensor_group_map,
        tensor_group_flag,
        tensor_group_idx,
        tensor_cache,
        tensor_batch_cnt,
        *tt_cores,
    ):
        ctx.learning_rate = learning_rate
        ctx.tt_p_shapes = tt_p_shapes
        ctx.tt_q_shapes = tt_q_shapes
        ctx.tt_ranks = tt_ranks
        ctx.tensor_p_shape = tensor_p_shape
        ctx.tensor_q_shape = tensor_q_shape
        ctx.tensor_tt_ranks = tensor_tt_ranks
        ctx.table_length = table_length
        ctx.feature_dim = feature_dim
        ctx.batch_cnt = batch_size
        ctx.tt_cores = tt_cores
        ctx.sorted_idx = sorted_idx
        ctx.sorted_key = sorted_key

        ctx.save_for_backward(
            indices,
            rowidx,
        )

        # breakpoint()
        output = Eff_TT_embedding_cuda.Eff_TT_bag_forward(
            batch_size,
            table_length, # need
            feature_dim, # need
            indices,
            offsets,
            rowidx,
            tt_p_shapes,
            tt_q_shapes,
            tt_ranks,
            tensor_p_shape,
            tensor_q_shape,
            tensor_tt_ranks,
            list(ctx.tt_cores),
            tensor_group_map,
            tensor_group_flag,
            tensor_group_idx,
            tensor_cache,
            tensor_batch_cnt
        )
        return output

    @staticmethod
    def backward(ctx, d_reduced: torch.Tensor)-> Tuple[torch.Tensor]:
        # raise NotImplementedError("Eff_TTEmbeddingBag is not implemented yet")
        
        
        (indices, rowidx) = ctx.saved_tensors

        if ctx.sorted_key == None:
            sorted_idx, sorted_key = indices.unique(sorted=True, return_inverse=True)
        else:
            sorted_idx = ctx.sorted_idx
            sorted_key = ctx.sorted_key
        # Eff_TT_embedding_cuda.Eff_TT_backward(
        # Eff_TT_embedding_cuda.Fused_Eff_TT_backward(
        Eff_TT_embedding_cuda.Fused_Extra_Eff_TT_bag_backward(
            ctx.batch_cnt,
            ctx.table_length,
            ctx.feature_dim,
            ctx.learning_rate,

            indices,
            rowidx,
            ctx.tt_p_shapes, 
            ctx.tt_q_shapes,
            ctx.tt_ranks, 
            ctx.tensor_p_shape,
            ctx.tensor_q_shape,
            ctx.tensor_tt_ranks,
            d_reduced,
            list(ctx.tt_cores),
            sorted_idx,
            sorted_key,
        )

        return tuple(
            [
                None,  
                None,  
                None,  
                None,
                None,  
                None,  
                None,  
                None,
                None,
                None,  
                None,  
                None,  
                None,  
                None,  
                None,  
                None,  
                None,  
                None,  
                None,
                None,
                None, # tt_core0
                None, # tt_core1
                None, # tt_core2
            ]
        )

class Eff_TTEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tt_ranks: List[int],
        tt_p_shapes: Optional[List[int]] = None,
        tt_q_shapes: Optional[List[int]] = None,
        optimizer: str = "SGD",
        learning_rate: float = 0.1,
        weight_dist: str = "normal",
        device=torch.device("cuda:0"),
        batch_size=4096, # Not used!
    ) -> None:
        super(Eff_TTEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tt_ranks = tt_ranks
        self.num_tt_core = len(self.tt_ranks) + 1
        self.tt_ranks = [1] + tt_ranks + [1]
        # self.batch_size = batch_size

        self.tt_p_shapes: List[int] = (
            suggested_tt_shapes(num_embeddings, self.num_tt_core)
            if tt_p_shapes is None
            else tt_p_shapes
        )
        self.tt_q_shapes: List[int] = (
            suggested_tt_shapes(embedding_dim, self.num_tt_core, False)
            if tt_q_shapes is None
            else tt_q_shapes
        )
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_dist = weight_dist
        self.device = device

        Eff_TT_embedding_cuda.init_cuda(
            device.index, 
            self.tt_p_shapes, 
            self.tt_q_shapes, 
            self.tt_ranks, 
            0, 
            embedding_dim
        )

        # init TT cores 
        self.tt_cores = torch.nn.ParameterList()
        for i in range(self.num_tt_core):
            self.tt_cores.append(
                torch.nn.Parameter(
                    torch.empty(
                        [
                            self.tt_p_shapes[i],
                            self.tt_ranks[i]
                            * self.tt_q_shapes[i]
                            * self.tt_ranks[i + 1],
                        ],
                        device=self.device,
                        dtype=torch.float32,
                    )
                )
            )
        # print(self.tt_cores[0].shape, self.tt_cores[1].shape, self.tt_cores[2].shape)

        self.reset_parameters()
        self.tensor_p_shape = torch.tensor(self.tt_p_shapes).to(self.device)
        self.tensor_q_shape = torch.tensor(self.tt_q_shapes).to(self.device)
        self.tensor_tt_ranks = torch.tensor(self.tt_ranks).to(self.device)

        cache_length = self.tt_p_shapes[0] * self.tt_p_shapes[1]
        cache_dim = self.tt_q_shapes[0] * self.tt_q_shapes[1] * self.tt_ranks[2]

        # self.register_buffer(
        #     "tensor_group_map",
        #     torch.empty((cache_length, ), dtype=torch.int64, device=self.device),
        # )
        # self.register_buffer(
        #     "tensor_group_flag",
        #     torch.empty((cache_length, ), dtype=torch.int32, device=self.device),
        # )
        # self.register_buffer(
        #     "tensor_group_idx", 
        #     torch.empty((1, ), dtype=torch.int32, device=self.device),
        # )
        # self.register_buffer(
        #     "tensor_cache",
        #     torch.empty((cache_length * cache_dim, ), dtype=torch.float32, device=self.device)
        # )
        
        # Eff_TT_embedding_cuda.check_init(
        #     device.index, 
        #     self.tt_p_shapes, 
        #     self.tt_q_shapes, 
        #     self.tt_ranks,
        #     0, 
        #     embedding_dim,
        #     self.tensor_group_map,
        #     self.tensor_group_flag,
        #     self.tensor_group_idx,
        #     self.tensor_cache,
        # )

        self.forward_gemm_compress_ratio_t = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.backward_gemm_compress_ratio_t = torch.tensor(0.0, dtype=torch.float32, device=self.device)
    
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
            var = np.prod(tt_ranks ** cr_exponent)
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
    
    def forward(self, indices, offsets=None, unique=None, inverse=None):
        if unique is None:
            unique, inverse = indices.unique(sorted=True, return_inverse=True)

        self.backward_gemm_compress_ratio_t.fill_(indices.numel() / unique.numel())

        if offsets is None:  # 
            # raise RuntimeError("Do not use Eff_TTEmbedding for non-bag case")
            batch_size = indices.numel()
            output = TT_core_function.apply(
                self.learning_rate,
                batch_size,
                self.num_embeddings,
                self.embedding_dim,
                indices,
                self.tt_p_shapes,
                self.tt_q_shapes,
                self.tt_ranks,
                self.tensor_p_shape,
                self.tensor_q_shape,
                self.tensor_tt_ranks,
                unique,
                inverse,
                self.tensor_group_map,
                self.tensor_group_flag,
                self.tensor_group_idx,
                self.tensor_cache,
                self.forward_gemm_compress_ratio_t,
                *(self.tt_cores),           
            )
            return output  
        else:
            # raise NotImplementedError("Eff_TTEmbeddingBag is not implemented yet")

            rowidx = Eff_TT_embedding_cuda.compute_rowidx(indices, offsets)


            cache_length = self.tt_p_shapes[0] * self.tt_p_shapes[1]
            cache_dim = self.tt_q_shapes[0] * self.tt_q_shapes[1] * self.tt_ranks[2]

            tensor_group_map = torch.empty((cache_length, ), dtype=torch.int64, device=offsets.device)
            tensor_group_flag = torch.empty((cache_length, ), dtype=torch.int32, device=offsets.device)
            tensor_group_idx = torch.empty((1, ), dtype=torch.int32, device=offsets.device)
            tensor_cache = torch.empty((cache_length * cache_dim, ), dtype=torch.float32, device=offsets.device)
 
            batch_cnt = indices.numel()
            output = TT_core_bag_function.apply(
                self.learning_rate,
                batch_cnt,
                self.num_embeddings,
                self.embedding_dim,
                indices,
                offsets,
                rowidx,
                self.tt_p_shapes,
                self.tt_q_shapes,
                self.tt_ranks,
                self.tensor_p_shape,
                self.tensor_q_shape,
                self.tensor_tt_ranks,
                unique,
                inverse,
                tensor_group_map,
                tensor_group_flag,
                tensor_group_idx,
                tensor_cache,
                self.forward_gemm_compress_ratio_t,
                *(self.tt_cores),           
            )

            return output



    

Eff_TTEmbeddingBag = Eff_TTEmbedding



