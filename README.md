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

## TF Datasets (Tensorflow-datasets)

| Dataset  | Config | Desc |
| ------------- | ------------- | ------------- | 
| klue | tc | KLUE benchmark의 topic classification task |
| klue | tc.full | tc에서 task 수행에 필요하지 않은 필드들이 추가된 configuration |
| klue | sts | Semnatic Texture Similarity task |
| klue | sts.full | sts에서 task 수행에 필요하지 않은 필드들이 추가된 configuration  |
| klue | nli | Natural Language Inference task |
| klue | nli.full | nli에서 task 수행에 필요하지 않은 필드들이 추가된 configuration |
| klue | ner | Named Entity Recognition task |
| klue | re | Relation Extraction task |
| klue | dp | Dependency Parsing task |
| klue | mrc | Machine Reading Comprehension task |
| klue | dst | Dialogue State Tracking task. Ontology는 dataset의 metadata로 저장되어 있음. |
| klue | dst.gen | Generative model을 위한 dst dataset. utterance마다 모든 슬롯들이 포함되어있기 때문에 (없는 경우 none으로 set), ontology는 따로 저장되어 있지 않음. |
| korquad | v1.0 | Korean Question Answering Dataset v1.0 task. 모든 질문에 대한 답이 context에 존재함. |
| korquad | v1.0.split | deterministic하게 train, validation, test split을 나눔. |
| korquad | v2.1 | Korean Question Answering Dataset v2.1 task. 모든 질문에 대한 답이 context에 존재함. |
| korquad | v2.1.split | deterministic하게 train, validation, test split을 나눔. |
| korquad | v2.1.html | context, answer이 html 형식임. |
| korquad | v2.1.html.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | newspaper.v1.0 | newspaper dataset. paragraph가 sentence들의 배열임 |
| nikl | newspaper.v1.0.page | sentence들을 하나의 paragraph로 merge함. |
| nikl | newspaper.v1.0.page.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | web.v1.0 | web dataset. document가 sentence들의 배열임 |
| nikl | web.v1.0.page | sentence들을 하나의 document로 merge함. |
| nikl | web.v1.0.page.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | web.v1.0.paragraph_page | example의 단위가 document가 아닌, paragraph임. |
| nikl | written.v1.0 | written dataset. document가 sentence들의 배열임 |
| nikl | written.v1.0.page | sentence들을 하나의 document로 merge함. |
| nikl | written.v1.0.page.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | written.v1.0.paragraph_page | example의 단위가 document가 아닌, paragraph임. |
| nikl | spoken.v1.0 | spoken dataset. nikl json 파일과 동일하게 그냥 parsing한 data |
| nikl | spoken.v1.0.utterance | example의 단위가 하나의 utterance와 dialogue history로 구성. 데이터 크기가 매우 큼 |
| nikl | spoken.v1.0.utterance.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | messenger.v1.0 | messenger dataset. nikl json 파일과 동일하게 그냥 parsing한 data |
| nikl | messenger.v1.0.utterance | example의 단위가 하나의 utterance와 dialogue history로 구성. 데이터 크기가 매우 큼 |
| nikl | messenger.v1.0.utterance.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | mp.v1.0 | 형태 분석 dataset. nikl json 파일과 동일하게 그냥 parsing한 data |
| nikl | mp.v1.0.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | ls.v1.0 | 어휘 의미분석 dataset. nikl json 파일과 동일하게 그냥 parsing한 data |
| nikl | ls.v1.0.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | ne.v1.0 | 개체명 분석 dataset. nikl json 파일과 동일하게 그냥 parsing한 data |
| nikl | ne.v1.0.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | dp.v1.0 | 구문 분석 dataset. nikl json 파일과 동일하게 그냥 parsing한 data |
| nikl | dp.v1.0.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | summarization.v1.0 | topic sentences와 summary sentences가 섞여있는 데이터. |
| nikl | summarization.v1.0.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | summarization.v1.0.summary | 사람이 요약한 요약문. |
| nikl | summarization.v1.0.summary.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | summarization.v1.0.topic | 사람이 topic 문장을 선택한 요약문. |
| nikl | summarization.v1.0.topic.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | paraphrase.v1.0 | 유사 문장 데이터 |
| nikl | paraphrase.v1.0.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | cola.v1.0 | 문법성 판단 데이터 |
| kor_corpora | nsmc | naver sentiment movie corpus task |
| kor_corpora | nsmc.split | deterministic하게 train, validation, test split을 나눔. |
| kor_corpora | qpair | question pair task. 두 question의 동일 여부 판단. |
| kor_corpora | qpair.split | deterministic하게 train, validation, test split을 나눔. |
| kor_corpora | kornli | KaKao Brain에서 번역한 NLI dataset |
| kor_corpora | kornli.split | deterministic하게 train, validation, test split을 나눔. |
| kor_corpora | korsts | KaKao Brain에서 번역한 STS dataset |
| kor_corpora | korsts.split | deterministic하게 train, validation, test split을 나눔. |
| kor_corpora | khsd | Korean Hate Speech dataset |
| kor_corpora | khsd.split | deterministic하게 train, validation, test split을 나눔. |


## HF Datasets (Huggingface-datasets)

| Dataset  | Config | Desc |
| ------------- | ------------- | ------------- |
| nikl | newspaper.v1.0 | newspaper dataset. paragraph가 sentence들의 배열임 |
| nikl | newspaper.v1.0.page | sentence들을 하나의 paragraph로 merge함. |
| nikl | newspaper.v1.0.page.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | web.v1.0 | web dataset. document가 sentence들의 배열임 |
| nikl | web.v1.0.page | sentence들을 하나의 document로 merge함. |
| nikl | web.v1.0.page.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | web.v1.0.paragraph_page | example의 단위가 document가 아닌, paragraph임. |
| nikl | written.v1.0 | written dataset. document가 sentence들의 배열임 |
| nikl | written.v1.0.page | sentence들을 하나의 document로 merge함. |
| nikl | written.v1.0.page.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | written.v1.0.paragraph_page | example의 단위가 document가 아닌, paragraph임. |
| nikl | spoken.v1.0 | spoken dataset. nikl json 파일과 동일하게 그냥 parsing한 data |
| nikl | spoken.v1.0.utterance | example의 단위가 하나의 utterance와 dialogue history로 구성. 데이터 크기가 매우 큼 |
| nikl | spoken.v1.0.utterance.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | messenger.v1.0 | messenger dataset. nikl json 파일과 동일하게 그냥 parsing한 data |
| nikl | messenger.v1.0.utterance | example의 단위가 하나의 utterance와 dialogue history로 구성. 데이터 크기가 매우 큼 |
| nikl | messenger.v1.0.utterance.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | mp.v1.0 | 형태 분석 dataset. nikl json 파일과 동일하게 그냥 parsing한 data |
| nikl | mp.v1.0.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | ls.v1.0 | 어휘 의미분석 dataset. nikl json 파일과 동일하게 그냥 parsing한 data |
| nikl | ls.v1.0.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | ne.v1.0 | 개체명 분석 dataset. nikl json 파일과 동일하게 그냥 parsing한 data |
| nikl | ne.v1.0.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | dp.v1.0 | 구문 분석 dataset. nikl json 파일과 동일하게 그냥 parsing한 data |
| nikl | dp.v1.0.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | summarization.v1.0 | topic sentences와 summary sentences가 섞여있는 데이터. |
| nikl | summarization.v1.0.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | summarization.v1.0.summary | 사람이 요약한 요약문. |
| nikl | summarization.v1.0.summary.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | summarization.v1.0.topic | 사람이 topic 문장을 선택한 요약문. |
| nikl | summarization.v1.0.topic.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | paraphrase.v1.0 | 유사 문장 데이터 |
| nikl | paraphrase.v1.0.split | deterministic하게 train, validation, test split을 나눔. |
| nikl | cola.v1.0 | 문법성 판단 데이터 |


## TODO

- [ ] Huggingface datasets KLUE benchmark, Kor_corpora dataset 생성
- [ ] NIKL DP, MP, LS ClassLabel로 변경.