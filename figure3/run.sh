# !/bin/bash
rm figure3/results.txt

for i in {0..30}
do
    python figure3/cache_vs_comp_eval.py --num_tt_models=$i >> figure3/results.txt
done
