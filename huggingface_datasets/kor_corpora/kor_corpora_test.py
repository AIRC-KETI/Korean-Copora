# Copyright 2021 san kim
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import datasets

KOR_CORPORA_TASK_SET = [
    "nsmc",
    "nsmc.split",
    "qpair",
    "qpair.split",
    "kornli",
    "kornli.split",
    "korsts",
    "korsts.split",
    "khsd",
    "khsd.split"
]


for kor_corpora_task in KOR_CORPORA_TASK_SET:
    dataset = datasets.load_dataset("kor_corpora.py", kor_corpora_task, cache_dir="../../cached_dir/huggingface_datasets")
    if 'train' in dataset:
        for data in dataset['train']:
            print(data)
            break
    else:
        first_key = list(dataset.keys())[0]
        for data in dataset[first_key]:
            print(data)
            break