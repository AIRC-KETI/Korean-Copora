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


import os
import csv
import json
import copy
import hashlib
import glob
import functools

import datasets

_DESCRIPTION = """
Description is **formatted** as markdown.
It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

_CITATION = """
"""

_VERSION = datasets.Version('1.0.0', "")

_DATASET_ROOT = {
    'mrc': 'AIHub/MRC/기계독해분야',
    'book-mrc': 'AIHub/BookMRC',
    'doc-summary': 'AIHub/DocSummary',
    'book-summary': 'AIHub/BookSummary',
    'paper-summary': 'AIHub/PaperSummary',
}

_MRC_QAS_SEQ_FEATURE = datasets.Sequence({
    'id': datasets.Value("string"),
    'question': datasets.Value("string"),
    'answerable': datasets.Value("bool"),
    'answers': datasets.Sequence({
        'answer_start': datasets.Value("int32"),
        'text': datasets.Value("string")
    }),
    'classtype': datasets.Value("string"),
})

_MRC_PARAGRAPHS_SEQ_FEATURE = datasets.Sequence({
    'context': datasets.Value("string"),
    'qas': _MRC_QAS_SEQ_FEATURE,
})

_MRC_FEATURE = datasets.Features({
    'idx': datasets.Value("int32"),
    'title': datasets.Value("string"),
    'paragraphs': _MRC_PARAGRAPHS_SEQ_FEATURE,
})

_BOOK_MRC_QAS_SEQ_FEATURE = datasets.Sequence({
    'question': datasets.Value("string"),
    'answers': datasets.Sequence({
        'answer_start': datasets.Value("int32"),
        'text': datasets.Value("string"),
    }),
    'id': datasets.Value("string"),
    'is_impossible': datasets.Value("bool"),
})

_BOOK_MRC_PARAGRAPHS_SEQ_FEATURE = datasets.Sequence({
    'context': datasets.Value("string"),
    'qas': _BOOK_MRC_QAS_SEQ_FEATURE,
})

_BOOK_MRC_FEATURE = datasets.Features({
    'time': datasets.Value("string"),
    'title': datasets.Value("string"),
    'agency': datasets.Value("string"),
    'year': datasets.Value("string"),
    'content_id': datasets.Value("string"),
    'KDC': datasets.Value("string"),
    'paragraphs': _BOOK_MRC_PARAGRAPHS_SEQ_FEATURE,
})

def _mrc_qas_proc(qas):
    result = list()
    for qa in qas:
        if "answers" in qa:
            answerable = True
            result.append({
                'id': qa['id'],
                'question': qa['question'],
                'answerable': answerable,
                'answers': qa['answers'],
                'classtype': qa['classtype']
            })
        else:
            answerable = False
            result.append({
                'id': qa['id'],
                'question': qa['question'],
                'answerable': answerable,
                'classtype': qa['classtype']
            })
    return result

def _mrc_paragraphs_proc(paragraphs):
    result = list()
    for paragraph in paragraphs:
        _context = paragraph['context']
        _qas = _mrc_qas_proc(paragraph['qas'])
        result.append({
            
            'context': _context,
            'qas': _qas
        })
    return result

def _parsing_mrc(file_path):
    with open(file_path, mode='r') as f:
        obj = json.load(f)
        idx = 0
        for doc in obj['data']:
            _title = doc['title']
            _paragraphs = _mrc_paragraphs_proc(doc['paragraphs'])

            yield idx, {
                'idx': idx,
                'title': _title,
                'paragraphs': _paragraphs,
            }

def _mrc_qas_noanswer_dic(qas):
    return {
        'id': qas['id'],
        'question': qas['question'],
        'answers': 'unanswerable',
    }

def _parsing_mrc_noanswer(file_path):
    with open(file_path, mode='r') as f:
        obj = json.load(f)
        for uid, doc in enumerate(obj['data']):
            _title = doc['title']
            _context = doc['paragraphs']['context']
            
            yield uid, {
                'idx': uid,
                'title': _title,
                'context': _context,
                'qas': _mrc_qas_noanswer_dic(doc['paragraphs']['qas']),
            }

def _book_mrc_qas_dic(qas):
    return {
        'id': qas['id'],
        'question': qas['question'],
        'answers': qas['answers']['text'],
        'is_impossible': qas['is_impossible']
    }

def _parsing_book_mrc(file_path):
    with open(file_path, mode='r') as f:
        obj = json.load(f.read())
        for uid, doc in enumerate(obj['data']):
            _title = doc['title']
            _context = doc['paragraphs']['context']
            
            yield uid, {
                'idx': uid,
                'title': _title,
                'context': _context,
                'qas': _book_mrc_qas_dic(doc['paragraphs']['qas']),
            }

def _hash_text(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def _filter_fn_hash_id(uid, split_fn):
    hash_id = _hash_text(str(uid))
    val = int(hash_id, 16)
    return split_fn(val)

_DEFAULT_RAW_CORPUS_SPLIT = {
              'source': [datasets.Split.TRAIN],
              'split': {
                datasets.Split.TRAIN: lambda x: x % 1000 > 0,
                datasets.Split.VALIDATION: lambda x: x % 1000 == 0,
              }}

_DEFAULT_DOWNSTREAMTASK_CORPUS_SPLIT = {
              'source': [datasets.Split.TRAIN],
              'split': {
                datasets.Split.TRAIN: lambda x: x % 10 > 1,
                datasets.Split.VALIDATION: lambda x: x % 10 == 0,
                datasets.Split.TEST: lambda x: x % 10 == 1,
              }}

class AIHubConfig(datasets.BuilderConfig):
    def __init__(self,
                 name,
                 data_root,
                 feature,
                 data_sp_path,
                 reading_fn,
                 parsing_fn,
                 additional_data_root=None,
                 homepage='https://aihub.or.kr/',
                 split_fn=None,
                 metadata=None,
                 **kwargs):
        super(AIHubConfig, self).__init__(
            name=name,
            version=_VERSION,
            **kwargs
        )
        self.data_root = data_root
        self.feature = feature
        self.data_sp_path = data_sp_path
        self.reading_fn = reading_fn
        self.parsing_fn = parsing_fn
        self.additional_data_root = additional_data_root
        self.homepage = homepage
        self.split_fn = split_fn
        self.metadata = metadata

class AIHub(datasets.GeneratorBasedBuilder):
    """DatasetBuilder for AIHub dataset."""

    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    BUILDER_CONFIGS = [
        AIHubConfig(
            name='mrc.normal.squad.v1.0',
            data_root=_DATASET_ROOT['mrc'],
            feature=_MRC_FEATURE,
            data_sp_path={datasets.Split.TRAIN: ['ko_nia_normal_squad_all.json']},
            reading_fn=_parsing_mrc,
            parsing_fn=lambda x:x,
        ),
        AIHubConfig(
            name='mrc.normal.squad.v1.0.split',
            data_root=_DATASET_ROOT['mrc'],
            feature=_MRC_FEATURE,
            data_sp_path={datasets.Split.TRAIN: ['ko_nia_normal_squad_all.json']},
            reading_fn=_parsing_mrc,
            parsing_fn=lambda x:x,
            split_fn=_DEFAULT_RAW_CORPUS_SPLIT,
        ),
        AIHubConfig(
            name='mrc.noanswer.squad.v1.0',
            data_root=_DATASET_ROOT['mrc'],
            feature=_MRC_FEATURE,
            data_sp_path={datasets.Split.TRAIN: ['ko_nia_noanswer_squad_all.json']},
            reading_fn=_parsing_mrc_noanswer,
            parsing_fn=lambda x:x,
        ),
        AIHubConfig(
            name='mrc.noanswer.squad.v1.0.split',
            data_root=_DATASET_ROOT['mrc'],
            feature=_MRC_FEATURE,
            data_sp_path={datasets.Split.TRAIN: ['ko_nia_noanswer_squad_all.json']},
            reading_fn=_parsing_mrc_noanswer,
            parsing_fn=lambda x:x,
            split_fn=_DEFAULT_RAW_CORPUS_SPLIT,
        ),
        AIHubConfig(
            name='mrc.clue0529.squad.v1.0',
            data_root=_DATASET_ROOT['mrc'],
            feature=_MRC_FEATURE,
            data_sp_path={datasets.Split.TRAIN: ['ko_nia_clue0529_squad_all.json']},
            reading_fn=_parsing_mrc,
            parsing_fn=lambda x:x,
        ),
        AIHubConfig(
            name='mrc.clue0529.squad.v1.0.split',
            data_root=_DATASET_ROOT['mrc'],
            feature=_MRC_FEATURE,
            data_sp_path={datasets.Split.TRAIN: ['ko_nia_clue0529_squad_all.json']},
            reading_fn=_parsing_mrc,
            parsing_fn=lambda x:x,
            split_fn=_DEFAULT_RAW_CORPUS_SPLIT,
        ),
        AIHubConfig(
            name='boock.mrc.v1.0',
            data_root=_DATASET_ROOT['mrc'],
            feature=_BOOK_MRC_FEATURE,
            data_sp_path={datasets.Split.TRAIN: ['Training/*.json'],
                          datasets.Split.VALIDATION: ['Validation/*.json']},
            reading_fn=_parsing_mrc,
            parsing_fn=lambda x:x,
        ),
    ]

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    For the NIKL, you must manually download NIKL data from https://aihub.or.kr/
    and extract it under the proper location.
    all the data have to located under manual_dir/AIHub.

    This is dataset and path pairs. (all the paths are case-sensitive!)
    ============================================
    MRC_NORMAL(v1.0): manual_dir/AIHub/MRC/기계독해분야/ko_nia_normal_squad_all.json
    MRC_NOANSWER(v1.0): manual_dir/AIHub/MRC/기계독해분야/ko_nia_noanswer_squad_all.json
    MRC_CLUE0529(v1.0): manual_dir/AIHub/MRC/기계독해분야/ko_nia_clue0529_squad_all.json
    BOOK_MRC(v1.0): manual_dir/AIHub/BookMRC/Training/도서.json
                    manual_dir/AIHub/BookMRC/Validation/도서.json
    ============================================
    """

    def _info(self) -> datasets.DatasetInfo:
        """Returns the dataset metadata."""
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=self.config.feature,
            homepage=self.config.homepage,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        """Returns SplitGenerators."""
        path_kv = {}
        for k, v in self.config.data_sp_path.items():
            path_list = []
            for vv in v:
                path_list.extend(glob.glob(os.path.join(
                    dl_manager.manual_dir, self.config.data_root, vv)))
            path_kv[k] = path_list
            path_list = []

        for _, v in path_kv.items():
            if len(v) == 0:
                raise AssertionError("For the AIHub dataset, you must manually download and extract dataset under {0}/{1}.".format(
                    dl_manager.manual_dir,
                    self.config.data_root
                ))

        if self.config.split_fn is not None:
            in_files = []
            for sp_s_key in self.config.split_fn['source']:
                in_files.extend(path_kv[sp_s_key])
            split_fn_kv = self.config.split_fn['split']
            return [
                datasets.SplitGenerator(name=k, gen_kwargs={'path_list': in_files, 'split_fn': v}) for k, v in split_fn_kv.items()
            ]

        return [
                datasets.SplitGenerator(name=k, gen_kwargs={'path_list': v}) for k, v in path_kv.items()
        ]

    def _generate_examples(self, path_list, split_fn=None):
        """Yields examples."""
        if split_fn is not None:
            split_filter = functools.partial(_filter_fn_hash_id, split_fn=split_fn)

        _hash_set = set()
        for file_path in path_list:
            try:
                for example in iter(self.config.reading_fn(file_path)):
                    uid, ex = self.config.parsing_fn(example)
                    
                    if split_fn is not None:
                        if not split_filter(str(uid)):
                            continue
                    hash_id = _hash_text(str(uid))
                    if hash_id not in _hash_set:
                        _hash_set.add(hash_id)
                        yield uid, ex
            except Exception as e:
                print(e)
