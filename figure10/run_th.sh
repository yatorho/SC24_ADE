#!/bin/bash

run_script=${1:-"figure10/e2e.py"}
local_batch_size=${2:-"4096"}
nnodes=${3:-8}
nprocs=${4:-4}
results_file=${5:-"figure10/results.txt"}


sp_dir="datasets/dlrm_pt/2022/splits"

# EcoRec benchmark
echo "EcoRec benchmark" > ${results_file}
for num_global_keys in 788 480 240
do
    echo "Dataset: Meta-${num_global_keys}: " >> ${results_file}

    torchrun --nnodes=${nnodes} --nproc-per-node=${nprocs} ${run_script} --launch=torch \
        --local_batch_size=${local_batch_size} --skew_degree=1 \
        --parallel=twp --model=EcoRec --num_micro_uidx=4 --reordering \
        --sp_dir=${sp_dir} --num_micro_keys=4 \
        --num_global_keys=${num_global_keys} --log_file="temp.txt"

    cat temp.txt >> ${results_file}
done


# EL-Rec benchmark
echo "====================================>> ${results_file}"
echo "EL-Rec benchmark" >> ${results_file}
for num_global_keys in 788 480 240
do
    echo "Dataset: Meta-${num_global_keys}: " >> ${results_file}

    torchrun --nnodes=${nnodes} --nproc-per-node=${nprocs} ${run_script} --launch=torch \
        --local_batch_size=${local_batch_size} --parallel=dp \
        --model=ELRec --sp_dir=${sp_dir} \
        --num_global_keys=${num_global_keys} --log_file="temp.txt"

    cat temp.txt >> ${results_file}
done


# TT-Rec benchmark
echo "====================================>> ${results_file}"
echo "EL-Rec benchmark" >> ${results_file}
for num_global_keys in 788 480 240
do
    echo "Dataset: Meta-${num_global_keys}: " >> ${results_file}

    torchrun --nnodes=${nnodes} --nproc-per-node=${nprocs} ${run_script} --launch=torch \
        --local_batch_size=${local_batch_size} --parallel=mp \
        --model=FBTT --sp_dir=${sp_dir} \
        --num_global_keys=${num_global_keys} --log_file="temp.txt"

    cat temp.txt >> ${results_file}
done
