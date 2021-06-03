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

NIKL_TASK_SET = [
    "newspaper.v1.0",
    "web.v1.0",
    "written.v1.0",
    "spoken.v1.0",
    "messenger.v1.0",
    "mp.v1.0",
    "ls.v1.0",
    "ne.v1.0",
    "dp.v1.0",
    "mp.v1.0.split",
    "ls.v1.0.split",
    "ne.v1.0.split",
    "dp.v1.0.split",
    "summarization.v1.0",
    "summarization.v1.0.split",
    "summarization.v1.0.summary",
    "summarization.v1.0.topic",
    "summarization.v1.0.summary.split",
    "summarization.v1.0.topic.split",
    "paraphrase.v1.0",
    "paraphrase.v1.0.split",
    "cola.v1.0",
    "ne.2020.v1.0",
    "ne.2020.v1.0.split",
    "cr.2020.v1.0",
    "cr.2020.full.v1.0",
]

dataset = datasets.load_dataset("nikl.py", "cr.2020.full.v1.0", data_dir="../../data", cache_dir="../../cached_dir/huggingface_datasets")
for data in dataset['train']:
    print(data)
    break

# for nikl_task in NIKL_TASK_SET:
#     dataset = datasets.load_dataset("nikl.py", nikl_task, data_dir="../../data", cache_dir="../../cached_dir/huggingface_datasets")
#     for data in dataset['train']:
#         print(data)
#         break
