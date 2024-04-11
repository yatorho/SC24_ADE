# !/bin/bash
rm table1/results.txt

python table1/mem_usage.py -B=10000 -M=FBTT >> table1/results.txt
python table1/mem_usage.py -B=20000 -M=FBTT >> table1/results.txt

python table1/mem_usage.py -B=10000 -M=PyTorch >> table1/results.txt
python table1/mem_usage.py -B=20000 -M=PyTorch >> table1/results.txt
