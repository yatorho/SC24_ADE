account=${1}
partition=${2}

local_batch_size=${3:-"4096"}
results_file=${4:-"figure12/results.txt"}
num_glboal_keys=${5:-"788"} # for Meta-788 dataset

sp_dir="datasets/dlrm_pt/2022/splits"
gpus_per_node=4

> ${results_file}
for nodes in 1 2 4 8
do
    # Nodes configuration
    echo "============ ${nodes}x${gpus_per_node}  benchmark ============" >> ${results_file}

    ## EcoRec benchmark
    echo "EcoRec benchmark: " >> ${results_file}
    srun --account=${account} --partition=${partition} --nodes=${nodes} \
        --gpus-per-node=${gpus_per_node} --ntasks-per-node=${gpus_per_node} --cpus-per-task=4 \
        python figure12/e2e.py --launch=slurm \
        --local_batch_size=${local_batch_size} --skew_degree=1 \
        --parallel=twp --model=EcoRec --num_micro_uidx=4 --reordering \
        --sp_dir=${sp_dir} --num_micro_keys=4 --sharding_method=greed \
        --num_global_keys=${num_global_keys} --log_file="temp.txt"
    cat temp.txt >> ${results_file}

    ## EL-Rec benchmark
    echo "EL-Rec benchmark: " >> ${results_file}
    srun --account=${account} --partition=${partition} --nodes=${nodes} \
        --gpus-per-node=${gpus_per_node} --ntasks-per-node=${gpus_per_node} --cpus-per-task=4 \
        python figure12/e2e.py --launch=slurm \
        --local_batch_size=${local_batch_size} --parallel=dp \
        --model=ELRec --sp_dir=${sp_dir} \
        --num_global_keys=${num_global_keys} --log_file="temp.txt"
    cat temp.txt >> ${results_file}

    ## TT-Rec benchmark
    echo "TT-Rec benchmark: " >> ${results_file}
    srun --account=${account} --partition=${partition} --nodes=${nodes} \
        --gpus-per-node=${gpus_per_node} --ntasks-per-node=${gpus_per_node} --cpus-per-task=4 \
        python figure12/e2e.py --launch=slurm \
        --local_batch_size=${local_batch_size} --parallel=mp \
        --model=FBTT --sp_dir=${sp_dir} --sharding_method=sequential \
        --num_global_keys=${num_global_keys} --log_file="temp.txt"
    cat temp.txt >> ${results_file}
done

