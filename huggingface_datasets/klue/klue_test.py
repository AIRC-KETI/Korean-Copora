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

# Error message: ValueError: External features info don't match the dataset:
# Got 'feature1'
# but expected something like 'feature2'
# the feature1 and the feature2 are exactly same without the order of keys
# builder config in 'datasets' has strict __eq__ function. so, we have to use functools.partial as attributes in custom builder configs.

# Error message: ValueError: Cannot name a custom BuilderConfig the same as an available BuilderConfig. Change the name.
# huggingface datasets save feature key after sorting as alphabet order. and compare the saved feature key in arrow file and the feature key in dataset info.
# if the order of feature key is different, even all the values and key pairs are same, the builder recognize it is different and then raise Value error

KLUE_TASK_SET = [
    "tc",
    "tc.full",
    "sts",
    "sts.full",
    "nli",
    "nli.full",
    "ner",
    "re",
    "dp",
    "mrc",
    "dst",
    "dst.gen"
]

for klue_task in KLUE_TASK_SET:
    dataset = datasets.load_dataset("klue.py", klue_task, cache_dir="../../cached_dir/huggingface_datasets")
    for data in dataset['train']:
        print(data)
        break

