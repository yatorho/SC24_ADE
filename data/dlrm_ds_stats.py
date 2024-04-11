import gzip
import io
import os
import time
import warnings
from typing import (Any, Dict, Iterable, Iterator, List, Optional, Tuple,
                    Union, cast)

import numpy as np
import torch
import torch.nn as nn
import torchrec
from torchrec import KeyedJaggedTensor

from ecorec.utils import iterate_in_groups

NUM_TABLES = {"2021": 856, "2022": 788}
BATCH_SIZES = {"2021": 65536, "2022": 131072}

def single_file_loader(path):
    print("Loading file: {}".format(path))
    with gzip.open(path) as f:
        indices, offsets, lengths = torch.load(f)  # type: ignore
        indices = indices.long()
        offsets = offsets.long()
        lengths = lengths

    num_tables = lengths.shape[0]
    batch_size = lengths.shape[1]

    keys = ["f{}".format(i) for i in range(num_tables)]
    features = torchrec.KeyedJaggedTensor(
        keys=keys,
        values=indices,
        offsets=offsets,
    )

    return features


def dlrm_datasets_compress(
    dlrm_dir,
    output_name,
    filters: Optional[List[Union[int, str]]] = None,
):
    """
    filters:
       None: means all files in dlrm_dir.
       [1, 2, 3, ...]: means dataset subscript by _ith.
       ["a", ith, ...]: means the datasets without subscription will be loaded.
    """
    # assert output_name is not None and cache_path is not None, "Either output_name or cache_path should be provided."
    dlrm_ds_id = dlrm_dir.split("/")[-1]

    indices_suffix = "indices_stats.txt"

    # gets paths in dir
    files = os.listdir(dlrm_dir)

    # filters
    files_to_remove = [file for file in files if not file.endswith(".pt.gz")]
    for rf in files_to_remove:
        files.remove(rf)

    if filters is not None:
        files_to_remove = []
        contain_wosubs = "a" in filters
        for file in files:
            d_th = file.split("_")[-1].split(".")[0]
            if d_th.startswith("bs"):
                if not contain_wosubs:
                    files_to_remove.append(file)
                continue

            d_th = int(d_th)
            if not d_th in filters:
                files_to_remove.append(file)

        for rf in files_to_remove:
            files.remove(rf)

    print("Loading from files: {}".format(files))
    feature_batches = [
        single_file_loader(os.path.join(dlrm_dir, file)) for file in files
    ]
    max_indices = [-1 for _ in range(NUM_TABLES[dlrm_ds_id])]
    min_indices = [0 for _ in range(NUM_TABLES[dlrm_ds_id])]

    for features in feature_batches:  # type: ignore
        for i, key in enumerate(features.keys()):
            indices = features[key].values()
            if indices.numel() > 0:
                max_indices[i] = max(max_indices[i], int(indices.max().item()))
                min_indices[i] = min(min_indices[i], int(indices.min().item()))

    if not os.path.exists(output_name):
        os.makedirs(output_name)

    with open(os.path.join(output_name, indices_suffix), "w") as f:
        f.write("combined dlrm datasets: {}\n".format(files))
        for i in range(NUM_TABLES[dlrm_ds_id]):
            f.write("f{}: [{}, {}]\n".format(i, min_indices[i], max_indices[i]))

    torch.save(
        {"files": files, "max_inds": max_indices, "min_indices": min_indices},
        os.path.join(output_name, "indices_stats.pt"),
    )

    for file, fs in zip(files, feature_batches):
        file_pt = file[:-3]

        print("Saving file: {}".format(file_pt))
        torch.save({"file": file_pt, "fs": fs}, os.path.join(output_name, file_pt))

    print("Done. Saved to: {}".format(output_name))


def _load_dlrm_ds_stats(ds_dir):
    indices_suffix = "indices_stats.pt"
    files_inds_dict = torch.load(os.path.join(ds_dir, indices_suffix))

    files: List[str] = files_inds_dict["files"]
    max_indices: list[int] = files_inds_dict["max_inds"]
    min_indices: list[int] = files_inds_dict["min_indices"]

    # filters
    files_to_remove = [file for file in files if not file.endswith(".pt.gz")]
    for rf in files_to_remove:
        files.remove(rf)

    return files, max_indices, min_indices


def dlrm_datasets_loader(
    ds_dir, filters: Optional[List[Union[int, str]]] = None, sort_files: bool = False
):
    """
    filters:
       None: means all files in dlrm_dir.
       [1, 2, 3, ...]: means dataset subscript by _ith.
       ["a", ith, ...]: means the datasets without subscription will be loaded.
    """
    print("Loading from dataset dir: {}".format(ds_dir))

    files, _, _ = _load_dlrm_ds_stats(ds_dir)
    files = sorted(files) if sort_files else files

    if filters is not None:
        files_to_remove = []
        contain_wosubs = "a" in filters
        for file in files:
            d_th = file.split("_")[-1].split(".")[0]
            if d_th.startswith("bs"):
                if not contain_wosubs:
                    files_to_remove.append(file)
                continue

            d_th = int(d_th)
            if not d_th in filters:
                files_to_remove.append(file)

        for rf in files_to_remove:
            files.remove(rf)

    # files.replace(".pt.gz", ".pt")
    files = [file[:-3] for file in files]

    print("Loading from files: {}".format(files))

    feature_batches = []

    for file in files:
        print("Loading file: {}".format(file))
        features_file_dict = torch.load(os.path.join(ds_dir, file))

        file_pt = features_file_dict["file"]
        assert file == file_pt, "File {} does not match file {} in dict".format(
            file, file_pt
        )

        features = features_file_dict["fs"]
        feature_batches.append(features)

    feature_batches = cast(List[torchrec.KeyedJaggedTensor], feature_batches)

    print("Load datasets done.")

    return feature_batches


def dlrm_datasets(
    ds_dir,
    filters: Optional[List[Union[int, str]]] = None,
    keys: Optional[List[str]] = None,
    batch_size: Optional[int] = None,
):
    feature_batches = dlrm_datasets_loader(ds_dir, filters=filters)
    # breakpoint()

    if keys is not None:
        keyed_fbs = []

        for fb in feature_batches:
            keyed_fb = torchrec.KeyedJaggedTensor.from_jt_dict(
                {key: fb[key] for key in keys if key in fb.keys()}
            )
            keyed_fbs.append(keyed_fb)
    else:
        keyed_fbs = feature_batches

    return keyed_fbs


def _gen_full_filters(files: List[str]) -> List[Union[int, str]]:
    wosubs_flag = "a"
    filters = []
    for file in files:
        d_th = file.split("_")[-1].split(".")[0]
        if d_th.startswith("bs"):
            filters.append(wosubs_flag)
            continue

        d_th = int(d_th)
        filters.append(d_th)

    return filters


def _save_chunk_list(key, chunk_lists, output_dir, chunk_size):
    ids = [min(chunk_lists[key][1]), max(chunk_lists[key][1])]
    ids = ",".join(list(map(str, ids)))
    chunk_obj_list = chunk_lists[key][0]
    chunk_list_path = os.path.join(
        output_dir, "{}_{}_[{}].pt".format(key, chunk_size, ids)
    )

    print(
        "Saving chunk list:[key: {}, len:{}, {}]".format(
            key, len(chunk_obj_list), chunk_list_path
        )
    )
    torch.save(chunk_obj_list, chunk_list_path)

    chunk_lists[key] = ([], [])


def _cat_group_indices(key, feature_batches, shuffle):
    cat_indices = torch.cat([fb[key].values() for fb in feature_batches])

    last_offsets = 0
    cat_offsets = [torch.tensor([0], device=cat_indices.device, dtype=torch.long)]
    for fb in feature_batches:
        offsets = fb[key].offsets()[1:] + last_offsets
        cat_offsets.append(offsets)
        last_offsets = offsets[-1]
    cat_offsets = torch.cat(cat_offsets)

    if shuffle:
        raise NotImplementedError("Shuffle not implemented yet.")

    return cat_indices, cat_offsets


def _update_chunk_stats(
    key, chunk_stats, i, chunk_size, chunk_indices, batch_size, group, filters
):
    chunk_stats[key]["id"] += 1
    chunk_stats[key]["local_id"] = i
    chunk_stats[key]["local_offsets"] = (
        i * chunk_size,
        (i + 1) * chunk_size,
    )
    chunk_stats[key]["offsets"] = (
        chunk_stats[key]["id"] * chunk_size,
        (chunk_stats[key]["id"] + 1) * chunk_size,
    )
    chunk_stats[key]["chunk_size"] = chunk_size
    chunk_stats[key]["num_indices"] = chunk_indices.numel()
    chunk_stats[key]["local_num_chunks"] = batch_size // chunk_size
    chunk_stats[key]["global_bach_size"] = batch_size // len(group) * len(filters)


def dlrm_datasets_splits(
    ds_dir,
    output_dir,
    filters: Optional[List[Union[int, str]]] = None,
    keys: Optional[List[str]] = None,
    chunk_size: int = 2048,
    shuffle: bool = True,
    group_size: int = 2,
    chunk_list_size=100,
):
    files, max_inds, min_inds = _load_dlrm_ds_stats(ds_dir)

    keys = ["f{}".format(i) for i in range(len(max_inds))] if keys is None else keys
    filters = sorted(
        _gen_full_filters(files) if filters is None else filters,
        key=lambda x: int(x) if isinstance(x, int) else -1,
    )

    num_indices_per_key = {key: max_inds[i] + 1 for i, key in enumerate(keys)}

    print("Split chunk from dataset dir: {} with filters: {}".format(ds_dir, filters))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        warnings.warn("Output dir: {} already exists.".format(output_dir))

    num_indices_save_name = "num_indices_per_key.pt"
    torch.save(num_indices_per_key, os.path.join(output_dir, num_indices_save_name))

    chunk_stats: Dict[str, Dict[str, Any]] = {}
    chunk_lists: Dict[str, Tuple[List, List]] = {}

    for group in iterate_in_groups(filters, group_size):
        feature_batches = dlrm_datasets_loader(ds_dir, filters=group, sort_files=True)

        if len(feature_batches) == 0:
            print("Skip invalid group: {}".format(group))
            continue

        batch_size = (feature_batches[0]["f0"].offsets().numel() - 1) * len(
            feature_batches
        )
        print(
            "Splitting {} batches with global batch size: {}, keys: {}, chunk_size: {}, group: {}".format(
                len(feature_batches), batch_size, keys, chunk_size, group
            )
        )

        for key in keys:
            print("Splitting key: {}".format(key))

            if key not in chunk_stats:
                chunk_stats[key] = {"id": -1}
            if key not in chunk_lists:
                chunk_lists[key] = ([], [])

            """
            #################### debug ####################
            batch_size = 6
            chunk_size = 3
            feature_batches = [] 
            feature_batches.append(torchrec.KeyedJaggedTensor(
                keys=["f0"],
                values=torch.LongTensor([9, 10, 9, 10]),
                offsets=torch.LongTensor([0, 2, 4]),
            ))
            feature_batches.append(torchrec.KeyedJaggedTensor(
                keys=["f0"],
                values=torch.LongTensor([9, 9, 10]),
                offsets=torch.LongTensor([0, 1, 3]),
            ))
            feature_batches.append(torchrec.KeyedJaggedTensor(
                keys=["f0"],
                values=torch.LongTensor([7, 8, 9]),
                offsets=torch.LongTensor([0, 2, 3]),
            ))
            ###############################################
            """

            cat_indices, cat_offsets = _cat_group_indices(key, feature_batches, shuffle)
            assert (
                cat_offsets[-1] == cat_indices.numel()
                and cat_offsets.numel() - 1 == batch_size
            )

            for i in range(batch_size // chunk_size):
                chunk_offsets = cat_offsets[i * chunk_size : (i + 1) * chunk_size + 1]
                chunk_indices = cat_indices[chunk_offsets[0] : chunk_offsets[-1]]
                chunk_offsets = chunk_offsets - chunk_offsets[0]

                _update_chunk_stats(
                    key,
                    chunk_stats,
                    i,
                    chunk_size,
                    chunk_indices,
                    batch_size,
                    group,
                    filters,
                )

                print(
                    "Extracting chunk: {}(chunk_id: {}), offsets: {}(glboal_offsets: {}), num_indices: {}".format(
                        i,
                        chunk_stats[key]["id"],
                        chunk_stats[key]["local_offsets"],
                        chunk_stats[key]["offsets"],
                        chunk_indices.numel(),
                    )
                )

                chunk = torchrec.KeyedJaggedTensor(
                    keys=[key],
                    values=chunk_indices,
                    offsets=chunk_offsets,
                )

                chunk_obj = {
                    "chunk": chunk,
                    "key": key,
                    "chunk_stats": chunk_stats[key],
                }

                chunk_lists[key][0].append(chunk_obj)
                chunk_lists[key][1].append(chunk_stats[key]["id"])

                if len(chunk_lists[key][0]) >= chunk_list_size:
                    _save_chunk_list(key, chunk_lists, output_dir, chunk_size)

    for key in keys:
        if len(chunk_lists[key][0]) > 0:
            _save_chunk_list(key, chunk_lists, output_dir, chunk_size)

    print("Done.")


def load_random_dlrm_datasets(keys, num_indices, batch_size, iter_num):
    raise NotImplementedError("load_random_dlrm_datasets not implemented yet.")


def _chunk_file_filter(
    chunk_lists: List[str],
    keys: List[str],
    num_samples: int,
    shuffle: bool,
    verbose: bool,
):
    def _chunk_list_generator(key):
        import re

        pattern = re.compile(r"(f\d+)_(\d+)_\[(\d+,\d+)\].pt")

        for chunk_list in chunk_lists:
            match = pattern.match(chunk_list)
            if match is not None and match.group(1) == key:
                yield chunk_list, int(match.group(2)), tuple(
                    map(int, match.group(3).split(","))
                )

    # sort by offsets
    keyed_chunk_list_names = {
        key: sorted(list(_chunk_list_generator(key)), key=lambda x: x[2][0])
        for key in keys
    }
    if shuffle:
        for key in keys:
            lst = keyed_chunk_list_names[key]
            np.random.shuffle(lst)
            keyed_chunk_list_names[key] = lst

    keyed_cum_samples = {}
    for key in keys:
        chunk_lst = keyed_chunk_list_names[key]

        new_chunk_lst = []
        cum_samples = 0
        print(
            "Sampling key: {} from {} chunks".format(key, len(chunk_lst))
        ) if verbose else None
        for chunk_name, chunk_size, chunk_ofs in chunk_lst:
            if num_samples >= 0 and cum_samples >= num_samples:
                break
            new_chunk_lst.append((chunk_name, chunk_size, chunk_ofs))

            chunk_num = chunk_size * (chunk_ofs[1] + 1 - chunk_ofs[0])
            cum_samples += chunk_num

        if num_samples >= 0 and cum_samples < num_samples:
            raise RuntimeError(
                "Not enough samples for key: {}[{}/{}]".format(
                    key, cum_samples, num_samples
                )
            )

        keyed_chunk_list_names[key] = new_chunk_lst
        keyed_cum_samples[key] = cum_samples

    # Check cum_samples is same for all keys
    cum_samples = list(keyed_cum_samples.values())
    assert len(set(cum_samples)) == 1, "cum_samples: {} are not same.".format(
        cum_samples
    )

    return keyed_chunk_list_names, cum_samples[0]


def _generate_keyed_chunk_map(keyed_chunk_list_names, batch_size, iter_num, verbose):
    def _gen_keyed_batched_ofs():
        keyed_batched_ofs: Dict[str, List[Tuple[List[str], Tuple[int, int]]]] = {
            key: [] for key in keyed_chunk_list_names.keys()
        }

        for key in keyed_chunk_list_names.keys():

            def _single_batch():
                batch_id = 0
                start_ofs = None
                cum_samples = 0
                batch_chunks = []

                for chunk_name, chunk_size, chunk_ofs in keyed_chunk_list_names[key]:
                    for i in range(chunk_ofs[1] + 1 - chunk_ofs[0]):
                        if start_ofs is None:
                            start_ofs = i
                        if chunk_name not in batch_chunks:
                            batch_chunks.append(chunk_name)

                        cum_samples += chunk_size

                        if cum_samples >= batch_size:
                            end_ofs = i

                            offsets = (start_ofs, end_ofs)
                            keyed_batched_ofs[key].append((batch_chunks, offsets))
                            batch_id += 1

                            cum_samples = 0
                            batch_chunks = []
                            start_ofs = None

                            if batch_id >= iter_num:
                                return

            _single_batch()

        return keyed_batched_ofs

    keyed_batched_ofs = _gen_keyed_batched_ofs()

    return keyed_batched_ofs


class DirDLRMDataset:
    def __init__(
        self,
        batch_size: int,
        keyed_chunk_map: Dict[str, List[Tuple[List[str], Tuple[int, int]]]],
        num_iters: int,
        sp_dir: str,
        num_indices: Dict[str, int],
        device: torch.device,
        verbose: bool = False,
    ):
        assert len(keyed_chunk_map.values()) > 0, "keyed_chunk_map is empty."
        assert (
            len(next(iter(keyed_chunk_map.values()))) == num_iters
        ), "num_iters: {} does not match the number of batches: {}".format(
            num_iters, len(next(iter(keyed_chunk_map.values())))
        )

        self.batch_size = batch_size
        self.keyed_chunk_map = keyed_chunk_map
        self.num_iters = num_iters
        self.sp_dir = sp_dir
        self.num_indices = num_indices
        self.verbose = verbose
        self.device = device

        self.keys = list(keyed_chunk_map.keys())

        self.reset()

    def reset(self, keys: Optional[List[int]] = None):
        if keys is None:
            self.indices_cache: Dict[int, KeyedJaggedTensor] = {}
        else:
            for key in keys:
                if key in self.indices_cache:
                    self.indices_cache.pop(key)

    def _load_chunk_list(self, file_name):
        full_path = os.path.join(self.sp_dir, file_name)
        obj = torch.load(full_path)
        return obj

    def _load_kjt_from_file(self, idx):
        indices = []
        lengths = []
        for key in self.keys:
            chunk_lists, (start_ofs, end_ofs) = self.keyed_chunk_map[key][idx]

            same_chunk_list = len(chunk_lists) == 1
            obj = self._load_chunk_list(chunk_lists[0])
            first_end_ofs = end_ofs + 1 if same_chunk_list else len(obj)
            start_kjt = [obj[i]["chunk"] for i in range(start_ofs, first_end_ofs)]

            if not same_chunk_list:
                obj = self._load_chunk_list(chunk_lists[-1])
                end_kjt = [obj[i]["chunk"] for i in range(end_ofs + 1)]
            else:
                end_kjt = []

            median_kjt = []
            for chunk_list in chunk_lists[1:-1]:
                obj = self._load_chunk_list(chunk_list)
                median_kjt.extend([obj[i]["chunk"] for i in range(len(obj))])

            key_cat_indices = torch.cat(
                [kjt.values() for kjt in start_kjt + median_kjt + end_kjt]
            )
            key_cat_lengths = torch.cat(
                [kjt.lengths() for kjt in start_kjt + median_kjt + end_kjt]
            )
            assert key_cat_lengths.numel() >= self.batch_size
            fixed_lengths = key_cat_lengths[: self.batch_size]
            fixed_indices = key_cat_indices[: fixed_lengths.sum().item()]

            assert fixed_lengths.numel() == self.batch_size
            assert fixed_lengths.sum().item() == fixed_indices.numel()

            indices.append(fixed_indices)
            lengths.append(fixed_lengths)

        indices = torch.cat(indices)
        lengths = torch.cat(lengths)

        kjt = torchrec.KeyedJaggedTensor(
            keys=self.keys,
            values=indices,
            lengths=lengths,
        )

        return kjt

    def cache_and_return(self, idx, kjt: KeyedJaggedTensor):
        self.indices_cache[idx] = kjt
        return kjt

    def load_kjt(self, idx: int):
        if idx in self.indices_cache:
            print("Loading from cache: {}".format(idx)) if self.verbose else None
            return self.indices_cache[idx]
        else:
            print("Loading from file: {}".format(idx)) if self.verbose else None
            kjt = self._load_kjt_from_file(idx).to(self.device)
            return self.cache_and_return(idx, kjt)

    def __len__(self):
        return self.num_iters

    def __getitem__(self, idx: int) -> KeyedJaggedTensor:
        if idx >= self.num_iters:
            raise IndexError("Index {} out of range.".format(idx))

        return self.load_kjt(idx)


def load_dir_dlrm_datasets(
    batch_size: int,
    sp_dir: str,
    keys: List[str],
    num_indices: Optional[Dict[str, int]] = None,
    iter_num: int = 10,
    shuffle: bool = False,
    device: torch.device = torch.device("cpu"),
    verbose: bool = False,
):
    num_indices_save_name = "num_indices_per_key.pt"
    # cast to absolute path
    sp_dir = os.path.abspath(sp_dir)
    _num_indices: Dict[str, int] = torch.load(
        os.path.join(sp_dir, num_indices_save_name)
    )

    if num_indices is not None:
        for key in num_indices.keys():
            assert (
                num_indices[key] <= _num_indices[key]
            ), "num_indices[{}] should be less than {}.".format(key, _num_indices[key])
    else:
        num_indices = _num_indices

    keys = ["f{}".format(i) for i in range(len(num_indices))] if keys is None else keys

    num_samples = batch_size * iter_num
    keyed_chunk_list_names, real_samples = _chunk_file_filter(
        os.listdir(sp_dir), keys, num_samples, shuffle, verbose
    )
    real_samples = real_samples if num_samples < 0 else num_samples
    real_num_iters = real_samples // batch_size

    keyed_chunk_map = _generate_keyed_chunk_map(
        keyed_chunk_list_names, batch_size, real_num_iters, verbose
    )
    dataset = DirDLRMDataset(
        batch_size,
        keyed_chunk_map,
        real_num_iters,
        sp_dir,
        num_indices,
        device,
        verbose,
    )
    return dataset


def keyed_dlrm_datasets_load(
    batch_size: int,
    keys: List[str],
    sp_dir: Optional[str] = None,
    num_indices: Optional[Dict[str, int]] = None,
    iter_num: int = 10,  # If use dir datasets, batches is max iters in sp_dir when iter_num is negative.
    shuffle: bool = False,
    use_random_datasets: bool = False,
    device: torch.device = torch.device("cpu"),
    verbose: bool = False,
):
    if use_random_datasets:
        assert (
            sp_dir is not None and num_indices is not None
        ), "sp_dir and num_indices should be provided when use_random_datasets is True."

        return load_random_dlrm_datasets(keys, num_indices, batch_size, iter_num)

    else:
        assert (
            sp_dir is not None
        ), "sp_dir should be provided when use_random_datasets is False."

        return load_dir_dlrm_datasets(
            batch_size, sp_dir, keys, num_indices, iter_num, shuffle, device, verbose
        )


class DenseDataset:
    def __init__(
        self,
        batch_size: int,
        dense_in_features: int,
        iter_num: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        verbose: bool = False,
    ):
        self.batch_size = batch_size
        self.dense_in_features = dense_in_features
        self.iter_num = iter_num
        self.device = device
        self.verbose = verbose
        self.dtype = dtype

        self.reset()

    def reset(self):
        self.indices_cache: Dict[int, torch.Tensor] = {}

    def _load_dense_data(self, idx):
        dense_t = torch.randn(self.batch_size, self.dense_in_features, dtype=self.dtype)
        return dense_t

    def cache_and_return(self, idx, dense_t: torch.Tensor):
        self.indices_cache[idx] = dense_t
        return dense_t

    def load_dense(self, idx: int):
        if idx in self.indices_cache:
            print("Loading from cache: {}".format(idx)) if self.verbose else None
            return self.indices_cache[idx]
        else:
            print(
                "Generate from dataset's device: {}".format(idx)
            ) if self.verbose else None
            dense_t = self._load_dense_data(idx).to(self.device)
            return self.cache_and_return(idx, dense_t)

    def __len__(self):
        return self.iter_num

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx >= self.iter_num:
            raise IndexError("Index {} out of range.".format(idx))

        return self.load_dense(idx)

class DLRMDataset:
    def __init__(
        self,
        batch_size: int,
        dense_in_features: int,
        sparse: DirDLRMDataset,
        iter_num = 10,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False
    ): 
        self.batch_size = batch_size
        self.dense_in_features = dense_in_features
        self.sparse = sparse
        self.iter_num = iter_num
        self.device = device
        self.verbose = verbose

    def __len__(self):
        return self.iter_num
    
    def __getitem__(self, idx: int):
        if idx >= self.iter_num:
            raise IndexError("Index {} out of range.".format(idx))

        sparse = self.sparse[idx]
        dense = torch.randn(self.batch_size, self.dense_in_features, device=self.device) 
        label = torch.randint(0, 2, (self.batch_size, 1), device=self.device)

        return sparse, dense, label

    def __iter__(self):
        for i in range(self.iter_num):
            yield self[i]


def random_dense_dlrm_dataset(
    batch_size: int,
    dense_in_features: int,
    iter_num: int,
    device: torch.device = torch.device("cpu"),
    verbose: bool = False,
):
    dataset = DenseDataset(
        batch_size, dense_in_features, iter_num, device, verbose=verbose
    )
    return dataset


def random_logits_dlrm_dataset(
    batch_size: int,
    iter_num: int,
    device: torch.device = torch.device("cpu"),
    verbose: bool = False,
):
    pass


def ring_dataset(datasets, num_iters):
    for i in range(num_iters):
        yield datasets[i % len(datasets)]


def sort_by_freq(
    input_tensor: torch.Tensor, return_resuffle=False, return_sorted=False
):
    uniques, counts = torch.unique(input_tensor, return_counts=True, sorted=True)
    uniques, counts = cast(torch.Tensor, uniques), cast(torch.Tensor, counts)

    sort_idx = torch.argsort(counts, descending=True)
    sorted_counts = counts[sort_idx]

    if return_sorted:
        sorted_values = uniques[sort_idx]
        sorted_tensor = sorted_values.repeat_interleave(sorted_counts)
    else:
        sorted_tensor = None

    if return_resuffle:
        resuffle = uniques.repeat_interleave(sorted_counts)
    else:
        resuffle = None

    return (uniques, sorted_counts), sorted_tensor, resuffle


def plot_cdf(
    feature_batches: List[torchrec.KeyedJaggedTensor],
    keys: Optional[List[str]] = None,
    file_name: Optional[str] = None,
    num_subplots_per_row=3,
    subplot_size=3,
    log_file: Optional[io.TextIOWrapper] = None,
    device=torch.device("cuda")
):
    import matplotlib.pyplot as plt

    feature_keys = keys if keys is not None else feature_batches[0].keys()

    num_feautures = len(feature_keys)
    num_rows = num_feautures // num_subplots_per_row + (
        1 if num_feautures % num_subplots_per_row != 0 else 0
    )
    print(
        "Figure size: {} x {} ({} x {} #n)".format(
            subplot_size * num_subplots_per_row,
            subplot_size * num_rows,
            num_subplots_per_row,
            num_rows,
        )
    )
    plt.figure(figsize=(subplot_size * num_subplots_per_row, subplot_size * num_rows))

    for i, key in enumerate(feature_keys):
        print("Ploting feature: {}".format(key))
        if log_file is not None:
            log_file.write("Ploting feature: {}\n".format(key))

        plt.subplot(num_rows, num_subplots_per_row, i + 1)
        cat_indices = torch.cat(
            [features[key].values() for features in feature_batches]
        ).to(device)

        def _iterplate(indices: torch.Tensor, freq: torch.Tensor, num_points=100):
            is_empty = indices.numel() == 0
            if is_empty:
                return np.array([]), np.array([])
            else:
                freq = freq.cumsum(0)
                freq = freq / freq[-1]

                # use linear interpolation to get `num_points` points
                uniqs_np, counts_np = indices.cpu().numpy(), freq.cpu().numpy()
                x = np.linspace(0, 1, num_points)
                counts_np = np.interp(x, uniqs_np / uniqs_np.max(), counts_np)
                return x, counts_np

        # plot X / D
        Ds = [1, 64, 128, 256, 512, 1024]
        for D in Ds:
            (indices, freq), _, _ = sort_by_freq(
                torch.div(cat_indices, D, rounding_mode="trunc")
            )
            max_id = -1 if indices.numel() == 0 else int(indices.max().item())

            plt.plot(
                *_iterplate(indices, freq), label="Max(X/{})={:.1e}".format(D, max_id)
            )
            plt.xlabel("Percentage of # indices")
            plt.ylabel("CDF")

            if log_file is not None:
                log_file.write(
                    "D: {}, Max(X/D): {}, U/B: {}/{}\n".format(
                        D, max_id, indices.numel(), freq.sum()
                    )
                )

        plt.title("{}".format(key))
        plt.legend()

    plt.tight_layout()
    save_file = (
        os.path.abspath(__file__)[:-3] + "tt_key_inds.png"
        if file_name is None
        else file_name
    )
    plt.savefig(save_file)
