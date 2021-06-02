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
    "v1.0",
    "v1.0.split",
    # "v2.1",
    # "v2.1.split",
    # "v2.1.html",
    # "v2.1.html.split"
]


for kor_corpora_task in KOR_CORPORA_TASK_SET:
    dataset = datasets.load_dataset("korquad.py", kor_corpora_task, cache_dir="../../cached_dir/huggingface_datasets")
    for data in dataset['train']:
        print(data)
        break