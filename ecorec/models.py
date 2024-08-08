import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torchrec.models.dlrm import DenseArch, InteractionArch, OverArch, choose

from .costs.sharding_plan import ShardingCost, ShardingPlan
from .tt_emb import tt_emb


class SimpleInteractionArch(nn.Module):
    """
    This is a simple interaction architecture that only concatenates the dense and sparse features for memory savings.
    """

    def __init__(self):
        super(SimpleInteractionArch, self).__init__()

    def forward(
        self, dense_features: torch.Tensor, sparse_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): an input tensor of size B X D.
            sparse_features (torch.Tensor): an input tensor of size B X F X D.

        Returns:
            torch.Tensor: an output tensor of size B X (F * D + D).
        """

        return torch.cat(
            [sparse_features.view(sparse_features.size(0), -1), dense_features], dim=1
        )


def create_embedding_bag(name, num_embeddings, embedding_dim, tt_ranks, learning_rate, device):
    if name == "FBTT":
        from ttrec import fbtembedding as fbte

        emb = fbte.FBTTEmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            tt_ranks=tt_ranks,
            enforce_embedding_dim=True,
            learning_rate=learning_rate,
            use_cache=False,
            cache_size=int(0.01 * num_embeddings),
            hashtbl_size=int(0.01 * num_embeddings),
        ).to(device)
    elif name == "ELRec":
        import elrec_ext.Efficient_TT.efficient_tt as elrec
        emb = elrec.Eff_TTEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            tt_ranks=tt_ranks,
            learning_rate=learning_rate,
            device=device
        ).to(device)
    elif name == "EcoRec":
        emb = tt_emb.TTEmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            tt_ranks=tt_ranks,
            enforce_embedding_dim=True,
            learning_rate=learning_rate,
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
        raise NotImplementedError

    return emb


class Model_Meta(nn.Module):
    def __init__(self):
        super(Model_Meta, self).__init__()

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "Model_Meta is an abstract class and should not be called directly."
        )


class TTEmbeddingLayer_Meta(Model_Meta):
    def __init__(
        self,
        batch_size: int,  # local batch_size
        keyed_num_embeddings: Dict[str, int],
        embedding_dim: int,
        tt_ranks: List[int],
        learning_rate: float = 0.001,
        keys: Optional[List[str]] = None, # global keys
        tt_work_keys: Optional[
            List[str]
        ] = None,  # decide which keys to use TT, if not given, all keys are used
        tt_emb: str = "EcoRec",  # 'EcoRec', 'FBTT', 'ELRec,
        device: torch.device = torch.device("cpu"),
        # Sharding plan
        sharding_plan: Optional[ShardingPlan] = None,
        sharding_cost: Optional[ShardingCost] = None,
    ):
        super(TTEmbeddingLayer_Meta, self).__init__()

        import torch.distributed as dist

        if dist.is_initialized():
            global_rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_rank = dist.get_rank() % world_size
        else:
            global_rank = 0
            world_size = 1
            local_rank = 0

        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size

        if keys is None:
            keys = list(keyed_num_embeddings.keys())
        else:
            for key in keys:
                assert key in keyed_num_embeddings, f"{key} not in keyed_num_embeddings."

        if sharding_plan is None or sharding_cost is None:
            if world_size != 1:
                raise ValueError(
                    "sharding_plan and sharding_cost must be provided for distributed training."
                )

            sharding_plan = {0: copy.deepcopy(keys)}
            sharding_cost = {0: 0.0}
        else:
            plan_keys = []
            for ks in sharding_plan.values():
                plan_keys.extend(ks)
            plan_keys = set(plan_keys)
            if plan_keys != set(keys):
                raise ValueError("global_keys in sharding_plan and `keys` must be same.")

        self.sharding_plan: ShardingPlan = sharding_plan
        self.sharding_cost: ShardingCost = sharding_cost

        self.global_keys = keys
        self.local_keys = self.sharding_plan[local_rank]

        self.local_batch_size = batch_size
        self.global_batch_size = batch_size * world_size

        self.keyed_num_embeddings = keyed_num_embeddings
        self.tt_work_keys = tt_work_keys if tt_work_keys is not None else keys
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.tt_ranks = tt_ranks
        self.device = device

        self.tt_emb = tt_emb


class TTEmbeddingLayer(nn.Module):
    def __init__(self, tt_embedding_layer_meta: TTEmbeddingLayer_Meta):
        super(TTEmbeddingLayer, self).__init__()
        self.embedding_meta = tt_embedding_layer_meta
        self.make_tt_embedding_layer()

    def make_tt_embedding_layer(self):
        embedding_meta = self.embedding_meta

        global_keys = embedding_meta.global_keys
        batch_size = embedding_meta.local_batch_size
        global_batch_size = embedding_meta.global_batch_size
        keyed_num_embeddings = embedding_meta.keyed_num_embeddings
        tt_work_keys = embedding_meta.tt_work_keys
        embedding_dim = embedding_meta.embedding_dim
        learning_rate = embedding_meta.learning_rate
        tt_ranks = embedding_meta.tt_ranks

        device = embedding_meta.device
        local_rank = embedding_meta.local_rank
        global_rank = embedding_meta.global_rank
        world_size = embedding_meta.world_size
        local_keys = embedding_meta.local_keys

        sharding_plan = embedding_meta.sharding_plan
        sharding_cost = embedding_meta.sharding_cost
        tt_emb = embedding_meta.tt_emb

        self.keyed_embs = {
            key: (
                create_embedding_bag(
                    tt_emb if key in tt_work_keys else "PyTorch",
                    keyed_num_embeddings[key],
                    embedding_dim,
                    tt_ranks,
                    learning_rate,
                    device,
                )
                if keyed_num_embeddings[key] != 0
                else None
            )
            for key in local_keys
        }


        self.mp_variables = {
            "global_keys": global_keys,
            "local_keys": local_keys,
            "keyed_embs": self.keyed_embs,
            "plan": sharding_plan,
            "embedding_dim": embedding_dim,
            "global_batch_size": global_batch_size,
        }


class MLP_Meta(Model_Meta):
    def __init__(
        self,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        num_sparse_features: int,
        over_arch_layer_sizes: List[int],
        embedding_dim: int,
        device: torch.device,
    ):
        super(MLP_Meta, self).__init__()

        if dense_arch_layer_sizes[-1] != embedding_dim:
            raise ValueError(
                f"dense_arch_layer_sizes[-1]({dense_arch_layer_sizes[-1]}) != embedding_dim({embedding_dim})"
            )

        self.dense_in_features = dense_in_features
        self.dense_arch_layer_sizes = dense_arch_layer_sizes
        self.num_sparse_features = num_sparse_features
        self.over_arch_layer_sizes = over_arch_layer_sizes
        self.embedding_dim = embedding_dim
        self.device = device


class MLP(nn.Module):
    def __init__(self, mlp_meta: MLP_Meta):
        super(MLP, self).__init__()
        self.mlp_meta = mlp_meta
        self.make_mlps()

    def make_mlps(self):
        dense_in_features = self.mlp_meta.dense_in_features
        dense_arch_layer_sizes = self.mlp_meta.dense_arch_layer_sizes
        num_sparse_features = self.mlp_meta.num_sparse_features
        over_arch_layer_sizes = self.mlp_meta.over_arch_layer_sizes
        embedding_dim = self.mlp_meta.embedding_dim
        device = self.mlp_meta.device

        self.dense_arch = DenseArch(
            in_features=dense_in_features,
            layer_sizes=dense_arch_layer_sizes,
            device=device,
        )

        # inter_arch = InteractionArch(
        #     num_sparse_features=num_sparse_features,
        # )
        self.inter_arch = SimpleInteractionArch()

        self.over_in_features: int = embedding_dim * (1 + num_sparse_features)

        self.over_arch = OverArch(
            in_features=self.over_in_features,
            layer_sizes=over_arch_layer_sizes,
            device=device,
        )

        self.mlp_variables = {
            "dense_arch": self.dense_arch,
            "inter_arch": self.inter_arch,
            "over_arch": self.over_arch,
        }
