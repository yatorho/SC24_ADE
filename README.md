# Accelerating Distributed DLRM Training with Optimized TT Decomposition and Micro-Batching

This repo is for SC 2024 artifacts evaluation.


## Environment Setup

### Prerequisites

+ CUDA 11.8 or later
+ torch 2.1.0+cu118
+ torchrec 0.5.0+cu118

You may encounter compatibility issues when using the `torcrec`, if your `torch` or `torchrec` version is different from the above. Please install the corresponding version of `torch`, `torchrec` and `fbgemm-gpu` by referring to the link below: 

```
https://download.pytorch.org/whl/cu<xxx>
```

, where `<xxx>` is the CUDA version you are using, such as `118` for CUDA 11.8, `121` for CUDA 12.1, etc.

### Installation Steps

Get the source code of EcoRec and unpack it to `your_path`. Run the following commands to install EcoRec:

```
cd {your_path}
pip install -e .
```

Download the source code of TT-Rec's kernel and install it according to the instructions in its `README.md` file:

```
git clone https://github.com/facebookresearch/FBTT-Embedding
# Follow the instructions in FBTT-Embedding/README.md
# Return back to {your_path}
```

Note: EL-Rec is integrated into the `elrec_ext` directory to accommodate specific functional expansions. Run the following command to install EL-Rec's kernel:

```
pip install elrec_ext/Efficient_TT
```

Before running the evaluation program, you need to manually append the project path to the environment variable `$PYTHONPATH`:

```
export PYTHONPATH=$PYTHONPATH:`pwd`
```

## Data Preparation

Make sure you have installed GIT-LFS(Git Large File Storage) before cloning the dataset repository and then download the meta DLRM datasets: 

```
mkdir datasets
git clone https://github.com/facebookresearch/dlrm_datasets.git
```

Preprocessing test data and the output will be generated in `datasets/dlrm_pt/2022/splits` directory:

```
python data/ds22_process.py
```

Note: Back to the root directory before running the above command.


## Running Experiment

We provide some scripts to run the experiments and get the results for our paper, including:

+ Figure 10:
    - We provide scripts to run the distributed DLRM training experiment.
    - Please also refer to `figure10/README.md` for more details.

+ Figure 11:
    - We provide scripts to evaluate the performance of TT-EMB's implementation of EcoRec, FBTT, and EL-Rec.
    - Please also refer to `figure11/README.md` for more details.
  
+ Figure 12:
    - We provide scripts to run the scaling efficiency experiment.
    - Please also refer to `figure12/README.md` for more details.

+ Figure 16:
    - We provide scripts to assess the efficiency of the table-wise pipeline model, reordering features, and slope feature counts strategy.
    - Please also refer to `figure16/README.md` for more details.
