# Evaluation for Figure 16 in the paper.

Our distributed experiments are implemented in a supercomputing center equipped with a `slurm` scheduling system.
Make sure there are four nodes available in your cluster, and each node has at least four GPUs.


You should first modify the first two lines in the `dlrm4x4.slurm` file:

```
#SBATCH --account={your_account}
#SBATCH --partition={your_partition}
```

, replacing `{your_account}` and `{your_partition}` with the account and computational partition of your cluster.

Then enter the root directory of the project, and run the following command:

```
figure16/run.sh
```

`run.sh` script will call sbatch to submit training tasks, and the results will be saved in `figure16/results.txt`. 

You can specify some parameters in `dlrm4x4.slrum` and `e2e.py` to customize the experiment:

+ `#SBATCH --nnodes`: Number of GPU nodes to use.

+ `#SBATCH --gpus-per-node`: Number of GPUs per node.

+ `--local_batch_size`: Batch size per GPU.

+ `--parallel`: Parallelism model, including 'dp' for Data Parallel, 'mp' for Model Parallel, 'bwp' for Bath-wise Pipeline and 'twp' for Table-wise Pipeline.

+ `--embedding_dim`: Embedding dimension.

+ `--dense_arch_layer_sizes`: Comma-separated list of bottom MLP sizes.

+ `--over_arch_layer_sizes`: Comma-separated list of top MLP sizes.

+ `--num_global_keys`: Number of features, i.e., table counts used.

+ `--sp_dir`: Datasets directory.

+ `--reordering`: Whether to enable reordering feature strategy  for table-wise pipeline model. 

+ `--skew_degree`: Skew degree for table-wise pipeline grain sharding, i.e., slope feature counts strategy . This param only works with table-wise pipeline model.

+ `--num_micro_keys`: Number of stages in pipeline scheduling. This param only works with table-wise pipeline model.

+ `--num_micro_uidx`: Number of micro-batches for micro-batching strategy. This param only works with table-wise pipeline model


