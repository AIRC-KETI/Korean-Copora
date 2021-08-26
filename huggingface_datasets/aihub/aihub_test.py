# Copyright 2021 hyeontae seo
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

dataset = datasets.load_dataset("aihub.py", "mrc.normal.squad.v1.0", data_dir="./data", cache_dir="./cached_dir/huggingface_datasets")

print(dataset["train"])
for idx, data in zip(range(10), dataset['train']):
    print(idx, data)
