import datasets
import hashlib
import functools
import glob
import os
import json
import openpyxl

_DESCRIPTION = """
"""

_CITATION = """
"""

_VERSION = datasets.Version('1.0.0', "")

_DATASET_ROOT = { # folder
    'common_squad': 'AIHub/common',
    'paper_summary': 'AIHub/논문자료 요약',
    'paper_patent_section': 'AIHub/논문자료 요약',
    'paper_patent_total': 'AIHub/논문자료 요약',
    'document_summary_law': 'AIHub/문서요약 텍스트',
    'document_summary_editorial': 'AIHub/문서요약 텍스트',
    'emotional_talk': 'AIHub/감성대화',
}

_PARAGRAPHS_SEQUENCE = datasets.Sequence({ # common_squad
    'qas': datasets.Sequence({
        'question': datasets.Value("string"),
        'answers': datasets.Sequence({
            'answer_start': datasets.Value("int32"),
            'text': datasets.Value("string"),
        }),
        'id': datasets.Value("string"),
    }),
    'context': datasets.Value("string"),
})

_COMMON_SQUAD_FEATURE = datasets.Features({ # common_squad
    'id': datasets.Value("int32"), 
    'paragraphs': _PARAGRAPHS_SEQUENCE,
    'title': datasets.Value("string"),
})


_SUMMARY_FEATRUE = datasets.Sequence({ # paper_summary 
    'orginal_text': datasets.Value("string"),
    'summary_text': datasets.Value("string"),
})

_PAPER_SUMMARY_FEATURE = datasets.Features({ # paper_summary
    'id': datasets.Value("int32"),  
    'doc_type': datasets.Value("string"),
    'doc_id': datasets.Value("string"),
    'title': datasets.Value("string"),
    'date': datasets.Value("string"), 
    'reg_no': datasets.Value("string"),  
    'ipc': datasets.Value("string"),  
    'issued_by': datasets.Value("string"),  
    'author': datasets.Value("string"), 
    'summary_entire': _SUMMARY_FEATRUE,
    'summary_section': _SUMMARY_FEATRUE,    
})

_PAPER_PATENT_SECTION_FEATURE = datasets.Features({ # paper_patent_section  
    'id': datasets.Value("int32"), 
    'doc_type': datasets.Value("string"),
    'doc_id': datasets.Value("string"),
    'title': datasets.Value("string"),
    'date': datasets.Value("string"), 
    'reg_no': datasets.Value("string"),  
    'ipc': datasets.Value("string"),  
    'author': datasets.Value("string"), 
    'summary_section': _SUMMARY_FEATRUE,    
})

_PAPER_PATENT_TOTAL_FEATURE = datasets.Features({ # paper_patent_total
    'id': datasets.Value("int32"), 
    'doc_type': datasets.Value("string"),
    'doc_id': datasets.Value("string"),
    'title': datasets.Value("string"),
    'date': datasets.Value("string"), 
    'reg_no': datasets.Value("string"),  
    'ipc': datasets.Value("string"), 
    'author': datasets.Value("string"), 
    'summary_entire': _SUMMARY_FEATRUE,   
    'summary_section': _SUMMARY_FEATRUE,  
})


# document_summary_law, document_summary_editorial, document_summary_newspaper
_TEXT_FEATURE = datasets.Sequence({ 
    'index': datasets.Value("int32"),
    'sentence': datasets.Value("string"),
    'highlight_indices': datasets.Value("string"),
})

# document_summary_law, document_summary_editorial, document_summary_newspaper
_DOCUMENT_QUALITY_SCORES = datasets.Features({
    'readable': datasets.Value("int32"),
    'accurate': datasets.Value("int32"),
    'informative': datasets.Value("int32"),
    'trustworthy': datasets.Value("int32"),
})

_DOCUMENT_SUMMARY_LAW_FEATURE = datasets.Features({ # document_summary_law
    'id': datasets.Value("string"),
    'category': datasets.Value("string"),
    'size': datasets.Value("string"),
    'char_count': datasets.Value("int32"),
    'publish_date': datasets.Value("string"),
    'title': datasets.Value("string"),
    'text': _TEXT_FEATURE, 
    'annotator_id': datasets.Value("int32"),
    'document_quality_scores': _DOCUMENT_QUALITY_SCORES,
    'extractive': datasets.Sequence(datasets.Value("int32")), 
    'abstractive': datasets.Sequence(datasets.Value("string")),
})

# document_summary_editorial, document_summary_newspaper
_DOCUMENT_SUMMARY_FEATURE = datasets.Features({ 
    'id': datasets.Value("string"),
    'category': datasets.Value("string"),
    'media_type': datasets.Value("string"),
    'media_sub_type': datasets.Value("string"),
    'media_name': datasets.Value("string"),
    'size': datasets.Value("string"),
    'char_count': datasets.Value("string"),
    'publish_date': datasets.Value("string"),
    'title': datasets.Value("string"),
    'text': _TEXT_FEATURE, 
    'annotator_id': datasets.Value("int32"),
    'document_quality_scores': _DOCUMENT_QUALITY_SCORES,  
    'extractive': datasets.Sequence(datasets.Value("int32")),
    'abstractive': datasets.Sequence(datasets.Value("string")),
})

_PERSONA_FEATURE = datasets.Features({ # emotional_talk
    'persona-id': datasets.Value("string"),
    'human': datasets.Sequence(
        datasets.Value("string"),
    ),
    'computer': datasets.Sequence(
        datasets.Value("string"),
    ),
})

_EMOTION_FEATURE = datasets.Features({ # emotional_talk
    'emotion-id': datasets.Value("string"),
    'type': datasets.Value("string"),
    'situation': datasets.Sequence(
        datasets.Value("string"),
    ),
})

_PROFILE_FEATURE = datasets.Features({ # emotional_talk
    'persona-id': datasets.Value("string"),
    'persona': _PERSONA_FEATURE,
    'emotion': _EMOTION_FEATURE,
})

_CONTENT_FEATURE = datasets.Features({ # emotional_talk
    'HS01': datasets.Value("string"),
    'SS01': datasets.Value("string"),
    'HS02': datasets.Value("string"),
    'SS02': datasets.Value("string"),
    'HS03': datasets.Value("string"),
    'SS03': datasets.Value("string"),
})

_TALK_FEATURE = datasets.Features({ # emotional_talk
    'id': datasets.Features({
        'profile-id': datasets.Value("string"),
        'talk-id': datasets.Value("string"),
    }),
    'content': _CONTENT_FEATURE,
})

_EMOTIONAL_TALK_FEATURE = datasets.Features({ # emotional_talk
    'id': datasets.Value("int32"), 
    'profile': _PROFILE_FEATURE,
    'talk': _TALK_FEATURE,
})



def _parsing_common_squad(file_path): # common_squad
    with open(file_path, mode='r') as f:
        obj = json.loads(f.read())

        for id, sample in enumerate(obj['data']):
            _id = id
            _paragraphs = sample['paragraphs']
            _title = sample['title']
            yield _id, {
                'id': _id,
                'paragraphs': _paragraphs,
                'title': _title,
            }

def _parsing_paper_summary(file_path): # paper_summary
    with open(file_path, mode='r') as f:
        obj = json.loads(f.read())

        for id, sample in enumerate(obj['data']):
            _id = id
            _doc_type = sample['doc_type']
            _doc_id = sample['doc_id']
            _title = sample['title']
            _date = sample['date']
            _reg_no = sample['reg_no']
            _ipc = sample['reg_no']
            _issued_by = sample['issued_by']
            _author = sample['author']
            _summary_entire = sample['summary_entire']
            _summary_section = sample['summary_section']
            yield _id, {
                'id': _id,
                'doc_type': _doc_type,
                'doc_id': _doc_id,
                'title': _title,
                'date': _date,
                'reg_no': _reg_no,
                'ipc': _ipc,
                'issued_by': _issued_by,
                'author': _author,
                'summary_entire': _summary_entire,
                'summary_section': _summary_section,
            } 


def _parsing_paper_patent_section(file_path): # paper_patent_section
    with open(file_path, mode='r') as f:
        obj = json.loads(f.read())

        for id, sample in enumerate(obj['data']):
            _id = id
            _doc_type = sample['doc_type']
            _doc_id = sample['doc_id']
            _title = sample['title']
            _date = sample['date']
            _reg_no = sample['reg_no']
            _ipc = sample['reg_no']
            _author = sample['author']
            _summary_section = sample['summary_section']
            yield _id, {
                'id': _id,
                'doc_type': _doc_type,
                'doc_id': _doc_id,
                'title': _title,
                'date': _date,
                'reg_no': _reg_no,
                'ipc': _ipc,
                'author': _author,
                'summary_section': _summary_section,
            }

        
def _parsing_paper_patent_total(file_path): # paper_patent_total
    with open(file_path, mode='r') as f:
        obj = json.loads(f.read())

        for id, sample in enumerate(obj['data']):
            _id = id
            _doc_type = sample['doc_type']
            _doc_id = sample['doc_id']
            _title = sample['title']
            _date = sample['date']
            _reg_no = sample['reg_no']
            _ipc = sample['reg_no']
            _author = sample['author']
            _summary_section = sample['summary_section']
            yield _id, {
                'id': _id,
                'doc_type': _doc_type,
                'doc_id': _doc_id,
                'title': _title,
                'date': _date,
                'reg_no': _reg_no,
                'ipc': _ipc,
                'author': _author,
                'summary_entire': _summary_section,
                'summary_section': _summary_section,
            } 


def _parsing_document_summary_law(file_path): # document_summary_law
    with open(file_path, mode='r') as f:
        obj = json.loads(f.read())

        for sample in obj:
            _id = sample['id']
            _category = sample['category']
            _size = sample['size']
            _char_count = sample['char_count']
            _publish_date = sample['publish_date']
            _title = sample['title']
            _text = sample['text']
            _annotator_id = sample['annotator_id']
            _document_quality_scores = sample['document_quality_scores']
            _extractive = sample['extractive']
            _abstractive = sample['abstractive']
            yield _id, {
                'id': _id,
                'category': _category,
                'size': _size,
                'char_count': _char_count,
                'publish_date': _publish_date,
                'title': _title,
                'text': _text,
                'annotator_id': _annotator_id,
                'document_quality_scores': _document_quality_scores, 
                'extractive': _extractive,
                'abstractive': _abstractive,
            }

# document_summary_editorial, document_summary_newspaper
def _parsing_document_summary(file_path):
    with open(file_path, mode='r') as f:
        obj = json.loads(f.read())

        for sample in obj:
            _id = sample['id']
            _category = sample['category']
            _media_type = sample['media_type']
            _media_sub_type = sample['media_sub_type']
            _media_name = sample['media_name']
            _size = sample['size']
            _char_count = str(sample['char_count']) 
            _publish_date = sample['publish_date']
            _title = sample['title']
            _text = sample['text']
            _annotator_id = sample['annotator_id']
            _document_quality_scores = sample['document_quality_scores']
            _extractive = sample['extractive']
            _abstractive = sample['abstractive']
            yield _id, {
                'id': _id,
                'category': _category,
                'media_type': _media_type,
                'media_sub_type': _media_sub_type,
                'media_name': _media_name,
                'size': _size,
                'char_count': _char_count,
                'publish_date': _publish_date,
                'title': _title,
                'text': _text,
                'annotator_id': _annotator_id,
                'document_quality_scores': _document_quality_scores, 
                'extractive': _extractive,
                'abstractive': _abstractive,
            }

def _parsing_emotional_talk(file_path): # emotional talk
    with open(file_path, mode='r') as f:
        obj = json.loads(f.read())

        for id, sample in enumerate(obj):
            _id = id
            _profile = sample['profile']
            _talk = sample['talk']
            yield _id, {
                'id':_id,
                'profile': _profile,
                'talk': _talk,
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
            name=name, # error...?
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
            name='common.squad.v1.0',
            data_root=_DATASET_ROOT['common_squad'],
            feature=_COMMON_SQUAD_FEATURE,
            data_sp_path={datasets.Split.TRAIN: ['nia_common_02_squad_질문, 답변, 제시문 말뭉치/ko_wiki_v1_squad.json']},
            reading_fn=_parsing_common_squad,
            parsing_fn=lambda x:x,
        ),
        AIHubConfig(
            name='common.squad.v1.0.split',
            data_root=_DATASET_ROOT['common_squad'],
            feature=_COMMON_SQUAD_FEATURE,
            data_sp_path={datasets.Split.TRAIN: ['nia_common_02_squad_질문, 답변, 제시문 말뭉치/ko_wiki_v1_squad.json']},
            reading_fn=_parsing_common_squad,
            parsing_fn=lambda x:x,
            split_fn=_DEFAULT_RAW_CORPUS_SPLIT,
        ),
        AIHubConfig(
            name='paper.summary.v1.0.split',
            data_root=_DATASET_ROOT['paper_summary'],
            feature=_PAPER_SUMMARY_FEATURE,
            data_sp_path={datasets.Split.TRAIN: ['Training/training_논문/*.json'],
                          datasets.Split.VALIDATION: ['Validation/validation_논문/*.json']},
            reading_fn=_parsing_paper_summary,
            parsing_fn=lambda x:x,
        ),
        AIHubConfig(
            name='paper.patent.section.v1.0.split',
            data_root=_DATASET_ROOT['paper_patent_section'],
            feature=_PAPER_PATENT_SECTION_FEATURE,
            data_sp_path={datasets.Split.TRAIN: ['Training/training_특허섹션만/*.json'],
                          datasets.Split.VALIDATION: ['Validation/validation_특허섹션만/*.json']},
            reading_fn=_parsing_paper_patent_section,
            parsing_fn=lambda x:x,
        ),
        AIHubConfig(
            name='paper.patent.total.v1.0.split',
            data_root=_DATASET_ROOT['paper_patent_total'],
            feature=_PAPER_PATENT_TOTAL_FEATURE,
            data_sp_path={datasets.Split.TRAIN: ['Training/training_특허전체/*.json'],
                          datasets.Split.VALIDATION: ['Validation/validation_특허전체/*.json']},
            reading_fn=_parsing_paper_patent_total,
            parsing_fn=lambda x:x,
        ),
        AIHubConfig(
            name='document.summary.law.v1.0.split',
            data_root=_DATASET_ROOT['document_summary_law'],
            feature=_DOCUMENT_SUMMARY_LAW_FEATURE,
            data_sp_path={datasets.Split.TRAIN: ['1.Training/train_법률_data/법률문서/train_original.json'],
                          datasets.Split.VALIDATION: ['2.Validation/valid_법률_data/법률문서/dev_original.json']},
            reading_fn=_parsing_document_summary_law,
            parsing_fn=lambda x:x,
        ),
        AIHubConfig(
            name='document.summary.editorial.v1.0.split',
            data_root=_DATASET_ROOT['document_summary_editorial'],
            feature=_DOCUMENT_SUMMARY_FEATURE,
            data_sp_path={datasets.Split.TRAIN: ['1.Training/train_사설잡지_data/train_original.json'],
                          datasets.Split.VALIDATION: ['2.Validation/valid_사설잡지_data/dev_original.json']},
            reading_fn=_parsing_document_summary,
            parsing_fn=lambda x:x,
        ),
        AIHubConfig(
            name='emotional.talk.v1.0.split',
            data_root=_DATASET_ROOT['emotional_talk'],
            feature=_EMOTIONAL_TALK_FEATURE,
            data_sp_path={datasets.Split.TRAIN: ['Training/감성대화말뭉치(최종데이터)_Training/감성대화말뭉치(최종데이터)_Training.json'],
                          datasets.Split.VALIDATION: ['Validation/감성대화말뭉치(최종데이터)_Validation/감성대화말뭉치(최종데이터)_Validation.json']},
            reading_fn=_parsing_emotional_talk,
            parsing_fn=lambda x:x,
        ),
        
    ]

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    For the NIKL, you must manually download NIKL data from https://aihub.or.kr/
    and extract it under the proper location.
    all the data have to located under manual_dir/AIHub.

    This is dataset and path pairs. (all the paths are case-sensitive!)
    ============================================
    COMMON_SQUAD(v1.0): manual_dir/AIHub/common/nia_common_02_squad_질문, 답변, 제시문 말뭉치/ko_wiki_v1_squad.json
    PAPER_SUMMARY(v1.0): manual_dir/AIHub/논문자료 요약/Training/training_논문/*.json
                         manual_dir/AIHub/논문자료 요약/Validation/validation_논문/*.json
    PAPER_PATENT_SECTION(v1.0): manual_dir/AIHub/논문자료 요약/Training/training_특허섹션만/*.json
                                manual_dir/AIHub/논문자료 요약/Validation/validation_특허섹션만/*.json
    PAPER_PATENT_TOTAL(v1.0): manual_dir/AIHub/논문자료 요약/Training/training_특허전체/*.json
                              manual_dir/AIHub/논문자료 요약/Validation/validation_특허전체/*.json
    DOCUMENT_SUMMARY_LAW(v1.0): manual_dir/AIHub/문서요약 텍스트/1.Training/train_법률_data/법률문서/train_original.json
                                manual_dir/AIHub/문서요약 텍스트/2.Validation/valid_법률_data/법률문서/dev_original.json
    DOCUMENT_SUMMARY_EDITORIAL(v1.0): manual_dir/AIHub/문서요약 텍스트/1.Training/train_사설잡지_data/train_original.json
                                      manual_dir/AIHub/문서요약 텍스트/2.Validation/valid_사설잡지_data/dev_original.json
    EMOTIONAL_TALK(v1.0): manual_dir/AIHub/감성대화/Training/감성대화말뭉치(최종데이터)_Training/감성대화말뭉치(최종데이터)_Training.json
                          manual_dir/AIHub/감성대화/Validation/감성대화말뭉치(최종데이터)_Validation/감성대화말뭉치(최종데이터)_Validation.json
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