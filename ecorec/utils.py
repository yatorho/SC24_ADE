import os
from typing import (Any, Dict, Iterable, Iterator, List, Optional, Sequence,
                    Tuple, TypeVar, Union, cast)

import torch
from torch import Tensor
from torch.cuda import cudart, nvtx
from torchrec import JaggedTensor, KeyedJaggedTensor


def iterate_in_groups(iter: Iterable[Any], group_size: int) -> Iterator[Any]:
    it = iter.__iter__()
    while True:
        chunk = []
        try:
            for _ in range(group_size):
                chunk.append(next(it))
        except StopIteration:
            if len(chunk) > 0:
                yield chunk
            break
        yield chunk

def partition_number(number: int, num_groups: int, skew_degree: float = 0, decreasing: bool = False) -> List[int]:
    if number < num_groups:
        raise ValueError(f"number must be greater than or equal to num_groups({number} vs. {num_groups})")
    if not (0 <= skew_degree <= 1):
        raise ValueError("skew_degree must be in the range [0, 1]")
    
    if num_groups == 1:
        return [number]
    
    def _desired_partition(n, g, slope):
        center_x = (g + 1) / 2
        center_y = n / g

        partition = [center_y + slope * (i - center_x) for i in range(1, g + 1)]

        if decreasing:
            partition.reverse()
        return partition

    k_max = 2 * (number - num_groups) / (num_groups * (num_groups - 1))
    desired_partition = _desired_partition(number, num_groups, k_max * skew_degree)

    partition = [1 for _ in range(0, num_groups)]
    
    # greedy algorithm
    for _ in range(0, number - num_groups):
        max_idx = max(range(0, num_groups), key=lambda i: desired_partition[i] - partition[i])
        partition[max_idx] += 1
    
    partition.sort(reverse=decreasing)

    return partition

T = TypeVar('T')
def iterate_in_num_groups(seq: Sequence[T], num_groups: int, slope_degree: float = 0, decreasing: bool = False):
    length = len(seq)
    partition = partition_number(length, num_groups, slope_degree, decreasing)

    start = 0
    for group_size in partition:
        yield seq[start: start + group_size]
        start += group_size


def nvtx_push(msg, cond=True):
    if cond:
        nvtx.range_push(msg)


def nvtx_pop(cond=True):
    if cond:
        nvtx.range_pop()


def interpret_dist(args):
    launch_method = args.launch

    if launch_method == "torch":
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]
    elif launch_method == "slurm":
        local_rank = int(os.environ["SLURM_LOCALID"])
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_NTASKS_PER_NODE"]
        )
        master_addr = os.environ["SLURM_SUBMIT_HOST"]
        master_port = "1234"
    else:
        raise NotImplementedError

    return world_size, rank, master_addr, master_port, local_rank

class DurationTimer:
    """
    Example:
        With DurationTimer() as t:
            ...
        duration = t.get_duration()
    """

    def __init__(self, cond=True, device=None, is_sync=True):
        self.cond = cond
        self.device = device if device is not None else torch.cuda.current_device()
        # self.device = device
        self.is_sync = is_sync

        self.duration = None

    def __enter__(self):
        if self.cond:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record() # type: ignore
        return self

    def __exit__(self, *args):
        if self.cond:
            self.end.record() # type: ignore
            if self.sync:
                torch.cuda.synchronize(self.device)
                self.duration = self.start.elapsed_time(self.end)

    def sync(self):
        if self.cond:
            torch.cuda.synchronize(self.device)
            self.duration = self.start.elapsed_time(self.end)
        return self

    def get_duration(self):
        if self.duration is not None:
            return self.duration
        else:
            return self.start.elapsed_time(self.end)


def lengths_to_offsets(lengths: Tensor) -> Tensor:
    return torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)  # type: ignore


def iterate_in_micro_batches(kjt: KeyedJaggedTensor, num_micro_batches):
    has_lengths = kjt.lengths_or_none() is not None
    # batch_size = kjt.lengths().numel() if has_lengths else kjt.offsets().numel() - 1
    batch_size = kjt.stride()

    assert batch_size % num_micro_batches == 0
    micro_batch_size = batch_size // num_micro_batches
    keys = kjt.keys()

    keyed_values = {key: kjt[key].values() for key in keys}
    if has_lengths:
        keyed_cumsum_length = {key: torch.zeros([], dtype=torch.int64, device=kjt.lengths().device) for key in keys}
        keyed_lengths = {key: kjt[key].lengths() for key in keys}
        for i in range(num_micro_batches):
            start_ofs = i * micro_batch_size
            end_ofs = start_ofs + micro_batch_size

            jt_dict: Dict[str, JaggedTensor] = {}

            for key in keys:
                micro_lengths = keyed_lengths[key][start_ofs:end_ofs]
                num_indices = micro_lengths.sum()
                cumsum_length = keyed_cumsum_length[key]

                micro_values = keyed_values[key][cumsum_length: cumsum_length + num_indices]
                keyed_cumsum_length[key] = cumsum_length + num_indices

                jt_dict[key] = JaggedTensor(values=micro_values, lengths=micro_lengths)

            yield KeyedJaggedTensor.from_jt_dict(jt_dict)
    else:
        keyed_offsets = {key: kjt[key].offsets() for key in keys}
        for i in range(num_micro_batches):
            start_ofs = i * micro_batch_size
            end_ofs = start_ofs + micro_batch_size

            jt_dict: Dict[str, JaggedTensor] = {}

            for key in keys:
                values_start_ofs = keyed_offsets[key][start_ofs]
                values_end_ofs = keyed_offsets[key][end_ofs]

                micro_offsets = keyed_offsets[key][start_ofs:end_ofs + 1] - values_start_ofs
                micro_values = keyed_values[key][values_start_ofs:values_end_ofs]

                jt_dict[key] = JaggedTensor(values=micro_values, offsets=micro_offsets)

            yield KeyedJaggedTensor.from_jt_dict(jt_dict)


def is_non_decreasing(tensor: torch.Tensor) -> bool:
    return bool((tensor == torch.sort(tensor).values).all().item())


def r0_print(*args, **kwargs):
    if torch.distributed.get_rank() == 0:
        print(*args, **kwargs)

if __name__ == "__main__":

    features = [f"f{i}" for i in range(20)]
    num_groups = 8

    for group in iterate_in_num_groups(features, num_groups, 0, decreasing=False):
        print(group)

    # kjt = KeyedJaggedTensor(
    #     keys=["f0", "f1"],
    #     values=torch.arange(20).unsqueeze(1),
    #     # lengths=torch.LongTensor([1, 2, 2, 5, 1, 2, 2, 5]),
    #     offsets = torch.LongTensor([0, 1, 3, 5, 10, 11, 13, 15, 20])
    # )
    # print(kjt.values().shape)

    # for micro_kjt in iterate_in_micro_batches(kjt, 2):
    #     print(micro_kjt.values())
    #     print(micro_kjt.lengths())
    #     print(micro_kjt.offsets())


