import torch.nn as nn

from .models import (MLP, MLP_Meta, Model_Meta, TTEmbeddingLayer,
                     TTEmbeddingLayer_Meta)


class DLRM_Meta(Model_Meta):
    def __init__(
        self,
        embedding_meta: TTEmbeddingLayer_Meta,
        mlp_meta: MLP_Meta,
    ):
        super(DLRM_Meta, self).__init__()

        self.embedding_meta = embedding_meta
        self.mlp_meta = mlp_meta


class DLRM(nn.Module):
    def __init__(
        self,
        dlrm_meta: DLRM_Meta,
    ):
        super(DLRM, self).__init__()
        self.dlrm_meta = dlrm_meta

        self.embeddings = TTEmbeddingLayer(dlrm_meta.embedding_meta)
        self.mlps = MLP(dlrm_meta.mlp_meta)
