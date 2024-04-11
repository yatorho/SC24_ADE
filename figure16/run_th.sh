#!/bin/bash

run_script=${1:-"figure16/e2e.py"}
local_batch_size=${2:-"16384"}
nnodes=${3:-1}
nprocs=${4:-4}
results_file=${5:-"figure16/results.txt"}


sp_dir="datasets/dlrm_pt/2022/splits"

num_global_keyss=100

# Batch-wise pipeline
echo "Batch-wise Pipeline" > ${results_file}
for num_global_keys in $num_global_keyss
do
    echo "Dataset: Meta-${num_global_keys}: " >> ${results_file}

    torchrun --nnodes=${nnodes} --nproc-per-node=${nprocs} ${run_script} --launch=torch \
        --local_batch_size=${local_batch_size} --skew_degree=1 \
        --parallel=bwp --model=EcoRec --num_micro_uidx=2 --reordering \
        --sp_dir=${sp_dir} --num_micro_batches=4 --sharding_method=greed \
        --num_global_keys=${num_global_keys} --log_file="temp.txt"

    cat temp.txt >> ${results_file}
done


# Table-wise pipeline
echo "==================================" >> ${results_file}
echo "Table-wise pipeline benchmark" >> ${results_file}
for num_global_keys in $num_global_keyss
do
    echo "Dataset: Meta-${num_global_keys}: " >> ${results_file}

    torchrun --nnodes=${nnodes} --nproc-per-node=${nprocs} ${run_script} --launch=torch \
        --local_batch_size=${local_batch_size} \
        --parallel=twp --num_micro_keys=4 --model=EcoRec --num_micro_uidx=2 \
        --sp_dir=${sp_dir} --num_micro_batches=4 --sharding_method=greed \
        --num_global_keys=${num_global_keys} --log_file="temp.txt"

    cat temp.txt >> ${results_file}
done


# Table-wise pipeline + Reordering
echo "==================================" >> ${results_file}
echo "Table-wise pipeline + Reordering benchmark" >> ${results_file}
for num_global_keys in $num_global_keyss
do
    echo "Dataset: Meta-${num_global_keys}: " >> ${results_file}

    torchrun --nnodes=${nnodes} --nproc-per-node=${nprocs} ${run_script} --launch=torch \
        --local_batch_size=${local_batch_size} \
        --parallel=twp --num_micro_keys=4 --model=EcoRec --num_micro_uidx=2 --reordering \
        --sp_dir=${sp_dir} --num_micro_batches=4 --sharding_method=greed \
        --num_global_keys=${num_global_keys} --log_file="temp.txt"

    cat temp.txt >> ${results_file}
done


# Table-wise pipeline + Reordering + Slope features
echo "==================================" >> ${results_file}
echo "Table-wise pipeline + Reordering + Slope features benchmark" >> ${results_file}
for num_global_keys in $num_global_keyss
do
    echo "Dataset: Meta-${num_global_keys}: " >> ${results_file}

    torchrun --nnodes=${nnodes} --nproc-per-node=${nprocs} ${run_script} --launch=torch \
        --local_batch_size=${local_batch_size} --skew_degree=1 \
        --parallel=twp --num_micro_keys=4 --model=EcoRec --num_micro_uidx=2 --reordering \
        --sp_dir=${sp_dir} --num_micro_batches=4 --sharding_method=greed \
        --num_global_keys=${num_global_keys} --log_file="temp.txt"

    cat temp.txt >> ${results_file}
done

