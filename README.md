# Accelerating Distributed DLRM Training with Optimized TT Decomposition and Micro-Batching

This repo is for SC 2024 artifacts evaluation.


## Environment Setup

Get sourc code of EcoRec and unpack it to `your_path`:

```
cd {your_path}
cd ecorec
```

Install EcoRec:

```
pip install .
```

Get source code of TT-Rec's kernel and install it:

```
git clone https://github.com/facebookresearch/FBTT-Embedding
cd FBTT-Embedding
pip install .
cd .. # back to the root directory
```

Note: we have integrated EL-Rec into `elrec_ext` directory for to accommodate certain functional expansions.


Before running the program, you need to manually append the project path to the environment variable `$PYTHONPATH`:

```
export PYTHONPATH=$PYTHONPATH:`pwd`
```

## Data Preparation

Download meta dlrm datasets:

```
mkdir datasets
git clone https://github.com/facebookresearch/dlrm_datasets.git
```

Preprocessing test data, and the output will be generated in `datasets/dlrm_pt/2022/splits` directory:

```
python data/ds22_process.py
```

Note: Back to the root directory before running the above command.


## Running Expriment

We provide some scripts to run the experiment and get the results for our paper, includes:

+ Figure 3:

    - We provide codes to nomalized lookup and memory compression rates with different number of TT-EMBs.

    - Please also refer to `figure3/README.md` for more details.

+ Figure 4:

    - We provide codes for redundancy analysis of DLRM datasets and TT computing pattern.

    - Please also refer to `figure4/README.md` for more details.


+ Figure 10:

    - We provide scripts to run the distributed DLRM training experiment.

    - Please also refer to `figure10/README.md` for more details.

+ Figure 11:

    - We provide scripts to evaluate the performance of TT-EMB's implementation of EcoRec, FBTT and EL-Rec.

    - Please also refer to `figure11/README.md` for more details.
  
+ Figure 12:

    - We provide scripts to run the scaling efficiency experiment.

    - Please also refer to `figure12/README.md` for more details.

+ Figure 16:

    - We provide scripts to access the efficiency of Table-wise pipeline model, reordering features and slope feature counts strategy.
    
    - Please also refer to `figure16/README.md` for more details.

