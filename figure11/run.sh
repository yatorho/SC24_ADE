
run_script=${1:-"figure11/tt_emb_perf.py"}
results_file=${2:-"figure11/results.txt"}


> ${results_file}
for batch_size in 2048 4096 8192 16384 32768 65536 131072
do
    echo "============ Batch size: ${batch_size} ================" >> ${results_file}
    python ${run_script} --batch_size=$batch_size --model=EcoRec >> ${results_file}
    python ${run_script} --batch_size=$batch_size --model=ELRec >> ${results_file}
    python ${run_script} --batch_size=$batch_size --model=FBTT >> ${results_file}
done


