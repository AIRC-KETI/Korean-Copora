# Korean-Copora


## Create TFDS
```bash
    pushd path_to_root_dir_of_dataset
    tfds build --data_dir path/to/directory --config dataset_configuration
    popd

    # e.g.
    # pushd tensorflow_datasets/klue
    # tfds build --data_dir ~/tensorflow_datasets --config ner
    # popd

    # More information
    # tfds build --help
```

## Create Huggingface dataset
```python
    import datasets
    dataset = datasets.load_dataset("path_to_dataset.py", "dataset_configuration", cache_dir="path/to/directory")

    # e.g.
    # dataset = datasets.load_dataset(
    #     "huggingface_datasets/nikl/nikl.py", 
    #     "summarization.v1.0.topic", 
    #     data_dir="/root/nfs3/ket5/data/raw_corpus/ko", 
    #     cache_dir="cached_dir/huggingface_datasets")
    # NIKL 데이터셋의 경우 자동 다운로드가 불가능하여 manual download를 해야함. 
    # manual download한 데이터를 formatting한 후 data_dir로 pass.
    # manual directory는 tfds의 경우 --manual_dir, huggingface datasets의 경우 data_dir
    # 데이터셋이 저장되는 directory는 tfds의 경우 --data_dir, huggingface datasets의 경우 cache_dir
```

## Dir Tree for NIKL

NIKL의 경우 아래 그림과 같이 데이터를 위치시킨 후 `manual_dir`을 argument로 pass해줘야 함.

```bash
manual_dir/
    └── NIKL
        └── v1.0
            ├── CoLA
            │   ├── NIKL_CoLA_in_domain_dev.tsv
            │   ├── NIKL_CoLA_in_domain_train.tsv
            │   ├── NIKL_CoLA_out_of_domain_dev.tsv
            │   └── References.tsv
            ├── DP
            │   └── NXDP1902008051.json
            ├── LS
            │   ├── NXLS1902008050.json
            │   └── SXLS1902008030.json
            ├── MESSENGER
            │   ├── MDRW1900000002.json
            │   ├── MDRW1900000003.json
            │   ├── MDRW1900000008.json
            │   ├── MDRW1900000010.json
            │   ├── MDRW1900000011.json
            │   ├── MDRW1900000012.json
                        .
                        .
                        .

```


## TODO

- [x] Huggingface datasets KLUE benchmark, Kor_corpora dataset 생성
- [ ] NIKL DP, MP, LS ClassLabel로 변경.