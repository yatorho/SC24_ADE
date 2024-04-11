from dlrm_ds_stats import dlrm_datasets_compress, dlrm_datasets_splits

if __name__ == "__main__":
    
    ds_dir = "datasets/dlrm_datasets/embedding_bag/2022"
    pt_dir = "datasets/dlrm_pt/2022/"
    sp_dir = "datasets/dlrm_pt/2022/splits"

    filters = ["a", 0, 1, 2, 3]
    keys = ["f0", "f1", "f3"]
    dlrm_datasets_compress(
        dlrm_dir=ds_dir,
        output_name=pt_dir,
        # filters=filters, # you can specify a subset for files
    )
    dlrm_datasets_splits(
        pt_dir,
        sp_dir,
        chunk_size=4096,
        # filters=filters, # you can specify a subset for files
        # keys=keys, # you can specify a subset for features
        shuffle=False,
        group_size=100,
    )
