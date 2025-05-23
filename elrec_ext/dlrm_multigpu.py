import torch
import torch.nn as nn
from torch._ops import ops
from torch.autograd.profiler import record_function
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
from torch.nn.parameter import Parameter
import numpy as np
import sys

### define dlrm in PyTorch ###
class DLRM_Net_multi_GPU(nn.Module):
    def create_mlp(self, ln, top=False):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, len(ln) - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if top and i == len(ln) - 2:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers
        # print(layers)
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln):
        emb_l = nn.ModuleList()
        for i in range(0, len(ln)):
            n = ln[i]

            EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
            # initialize embeddings
            # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
            W = np.random.uniform(
                low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
            ).astype(np.float32)
            # approach 1
            EE.weight.data = torch.tensor(W, requires_grad=True)
            # approach 2
            # EE.weight.data.copy_(torch.tensor(W))
            # approach 3
            # EE.weight = Parameter(torch.tensor(W),requires_grad=True)
            emb_l.append(EE)
        return emb_l

    def __init__(
        self,
        dim_feature=None,
        list_emb=None,
        list_bot=None,
        list_top=None,
        arch_interaction_op=None, #'dot'
        nDev=1
    ):
        super(DLRM_Net_multi_GPU, self).__init__()

        # save arguments
        self.arch_interaction_op = arch_interaction_op

        self.ndevices = nDev
        self.output_d = 0
        self.parallel_model_batch_size = -1
        self.parallel_model_is_not_prepared = True
        self.sync_dense_params = True

        # create operators
        self.emb_l = self.create_emb(dim_feature, list_emb)
        self.bot_l = self.create_mlp(list_bot).to("cuda:0")
        self.top_l = self.create_mlp(list_top, top=True).to("cuda:0")
        self.loss_fn = torch.nn.BCELoss(reduction="mean")
          

    def apply_mlp(self, x, layers):

        # approach 1: use ModuleList
        # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers
        return layers(x)

    def apply_emb(self, lS_i, emb_l):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups
        ly = []
        # print(lS_o.shape,lS_i.shape)
        # length = lS_i[0].shape[0]
        # sparse_offset_group_batch = torch.tensor(range(lS_i[0].shape[0])).to(lS_i.device)
        # print(lS_i[0].shape[0])
        for k, sparse_index_group_batch in enumerate(lS_i):
            E = emb_l[k]
            # _shape = sparse_index_group_batch.shape
            # _input = sparse_index_group_batch.reshape(length,1)
            
            V = E(
                sparse_index_group_batch
                # sparse_index_group_batch,
                # sparse_offset_group_batch,
            )
            ly.append(V)

        # print(ly)
        return ly

    def interact_features(self, x, ly):

        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            _, ni, nj = Z.shape
            offset = 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R

    def forward(self, dense_x, lS_i):
        # return self.sequential_forward(dense_x, lS_i)
        return self.parallel_forward(dense_x, lS_i)

    def parallel_forward(self, dense_x, lS_i):
        batch_size = dense_x.size()[0]
        ndevices = min(self.ndevices, batch_size, len(self.emb_l))
        device_ids = range(ndevices)
        if self.parallel_model_batch_size != batch_size:
            self.parallel_model_is_not_prepared = True

        if self.parallel_model_is_not_prepared or self.sync_dense_params:
            # replicate mlp (data parallelism)
            self.bot_l_replicas = replicate(self.bot_l, device_ids)
            self.top_l_replicas = replicate(self.top_l, device_ids)
            self.parallel_model_batch_size = batch_size

        if self.parallel_model_is_not_prepared:
            # distribute embeddings (model parallelism)
            t_list = []
            w_list = []
            for k, emb in enumerate(self.emb_l):
                d = torch.device("cuda:" + str(k % ndevices))
                t_list.append(emb.to(d))
                w_list.append(None)
            self.emb_l = nn.ModuleList(t_list)
            self.v_W_l = w_list
            self.parallel_model_is_not_prepared = False


        dense_x = scatter(dense_x, device_ids, dim=0)


        i_list = []
        for k, _ in enumerate(self.emb_l):
            d = torch.device("cuda:" + str(k % ndevices))
            i_list.append(lS_i[k].to(d))
        lS_i = i_list
      
        x = parallel_apply(self.bot_l_replicas, dense_x, None, device_ids)
  
        ly = self.apply_emb(lS_i, self.emb_l)
       
        if len(self.emb_l) != len(ly):
            sys.exit("ERROR: corrupted intermediate result in parallel_forward call")

        t_list = []
        for k, _ in enumerate(self.emb_l):
            d = torch.device("cuda:" + str(k % ndevices))
            y = scatter(ly[k], device_ids, dim=0)
            t_list.append(y)
        # adjust the list to be ordered per device
        ly = list(map(lambda y: list(y), zip(*t_list)))
      
        z = []
        for k in range(ndevices):
            zk = self.interact_features(x[k], ly[k])
            z.append(zk)
      
        p = parallel_apply(self.top_l_replicas, z, None, device_ids)

        p0 = gather(p, self.output_d, dim=0)

  
        z0 = p0

        return z0

    def sequential_forward(self, dense_x, lS_i):
        # process dense features (using bottom mlp), resulting in a row vector
        x = self.apply_mlp(dense_x, self.bot_l)

        # process sparse features(using embeddings), resulting in a list of row vectors
        ly = self.apply_emb(lS_i, self.emb_l)

        # interact features (dense and sparse)
        z = self.interact_features(x, ly)

        # obtain probability of a click (using top mlp)
        p = self.apply_mlp(z, self.top_l)

        return p

    def print_emb_footprint(self):
        i = 0
        _sum = 0
        for table in self.emb_l:
            size = table.embedding_dim * table.num_embeddings * 4 # float32
            i += 1
            _sum += size
        print("sum:",_sum, "~:",_sum/1024/1024,"MB")