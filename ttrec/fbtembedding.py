import torch

from . import tt_embeddings_ops as tt_ops
from .tt_embeddings_ops import OptimType


class FBTTEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        tt_ranks,
        tt_q_shapes=None,
        tt_p_shapes=None,
        enforce_embedding_dim=False,
        atomic_grad_acc=True,  # just for API compatibility
        optimizer: OptimType = OptimType.SGD,
        learning_rate: float = 0.001,
    ):
        super(FBTTEmbedding, self).__init__()
        self.backend = tt_ops.TTEmbeddingBag(
            num_embeddings,
            embedding_dim,
            tt_ranks,
            tt_q_shapes,
            tt_p_shapes,
            optimizer=optimizer,
            learning_rate=learning_rate,
            enforce_embedding_dim=enforce_embedding_dim,
            weight_dist='normal',
            sparse=True,
            use_cache=False
        )

    def forward(self, indices):
        return self.backend(indices, torch.arange(
            0, 
            indices.size(-1) + 1,
            dtype=torch.int64,
            device=indices.device
        ))


class FBTTEmbeddingBag(torch.nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        tt_ranks,
        tt_q_shapes=None,
        tt_p_shapes=None,
        enforce_embedding_dim=False,
        atomic_grad_acc=True,  # just for API compatibility
        optimizer: OptimType = OptimType.SGD,
        learning_rate: float = 0.001,
        use_cache=False,
        cache_size = 0,
        hashtbl_size = 0,
        num_micros=0, # invalid number represents to use default strategy, i.e., use batch_count = 20000
    ):
        super(FBTTEmbeddingBag, self).__init__()
        self.backend = tt_ops.TTEmbeddingBag(
            num_embeddings,
            embedding_dim,
            tt_ranks,
            tt_p_shapes,
            tt_q_shapes,
            optimizer=optimizer,
            learning_rate=learning_rate,
            enforce_embedding_dim=enforce_embedding_dim,
            weight_dist='normal',
            sparse=True,
            use_cache=use_cache,
            cache_size=int(cache_size),
            hashtbl_size=int(hashtbl_size)
        )

        self.num_micros = num_micros

    def forward(self, indices, offsets):
        if self.num_micros <= 0:
            return self.backend(
                indices, 
                offsets, 
            )
        else:
            batch_count = indices.size(0) // self.num_micros
            return self.backend(
                indices, 
                offsets, 
                batch_count=batch_count
            )

    def cache_populate(self):
        self.backend.cache_populate()
    
    def last_cache_hit_ratio(self):
        return self.backend.last_cache_hit_rate

