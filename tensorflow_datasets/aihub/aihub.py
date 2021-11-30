import hashlib
import functools
import glob
import os
import json

from datasets import builder

import tensorflow as tf
import tensorflow_datasets as tfds

_DESCRIPTION = """
"""

_CITATION = """
"""

_VERSION = tfds.core.Version('1.0.0', "")

_DATASET_ROOT = { # folder
    'common_squad': 'AIHub/common',
    'paper_summary': 'AIHub/논문자료 요약',
    'paper_patent_section': 'AIHub/논문자료 요약',
    'paper_patent_total': 'AIHub/논문자료 요약',
    'document_summary_law': 'AIHub/문서요약 텍스트',
    'document_summary_editorial': 'AIHub/문서요약 텍스트',
    'emotional_talk': 'AIHub/감성대화',
    'dialog': 'AIHub/dialog',
    'dialog_intent': 'AIHub/dialog',
    'dialog_headword': 'AIHub/dialog',
    'dialog_knowledge': 'AIHub/dialog'
}

_PARAGRAPHS_SEQUENCE = tfds.features.Sequence({ # common_squad
    'qas': tfds.features.Sequence({
        'question': tfds.features.Text(),
        'answers': tfds.features.Sequence({
            'answer_start': tf.int32,
            'text': tfds.features.Text(),
        }),
        'id': tfds.features.Text(),
    }),
    'context': tfds.features.Text(),
})

_COMMON_SQUAD_FEATURE = tfds.features.FeaturesDict({ # common_squad
    'id': tf.int32,
    'paragraphs': _PARAGRAPHS_SEQUENCE,
    'title': tfds.features.Text(),
})


_SUMMARY_FEATRUE = tfds.features.Sequence({ # paper_summary 
    'orginal_text': tfds.features.Text(),
    'summary_text': tfds.features.Text(),
})

_PAPER_SUMMARY_FEATURE = tfds.features.FeaturesDict({ # paper_summary
    'id': tf.int32,  
    'doc_type': tfds.features.Text(),
    'doc_id': tfds.features.Text(),
    'title': tfds.features.Text(),
    'date': tfds.features.Text(), 
    'reg_no': tfds.features.Text(),  
    'ipc': tfds.features.Text(),  
    'issued_by': tfds.features.Text(),  
    'author': tfds.features.Text(), 
    'summary_entire': _SUMMARY_FEATRUE,
    'summary_section': _SUMMARY_FEATRUE,    
})

_PAPER_PATENT_SECTION_FEATURE = tfds.features.FeaturesDict({ # paper_patent_section  
    'id': tf.int32, 
    'doc_type': tfds.features.Text(),
    'doc_id': tfds.features.Text(),
    'title': tfds.features.Text(),
    'date': tfds.features.Text(), 
    'reg_no': tfds.features.Text(),  
    'ipc': tfds.features.Text(),  
    'author': tfds.features.Text(), 
    'summary_section': _SUMMARY_FEATRUE,    
})

_PAPER_PATENT_TOTAL_FEATURE = tfds.features.FeaturesDict({ # paper_patent_total
    'id': tf.int32, 
    'doc_type': tfds.features.Text(),
    'doc_id': tfds.features.Text(),
    'title': tfds.features.Text(),
    'date': tfds.features.Text(), 
    'reg_no': tfds.features.Text(),  
    'ipc': tfds.features.Text(), 
    'author': tfds.features.Text(), 
    'summary_entire': _SUMMARY_FEATRUE,   
    'summary_section': _SUMMARY_FEATRUE,  
})


# document_summary_law, document_summary_editorial, document_summary_newspaper
_TEXT_FEATURE = tfds.features.Sequence({ 
    'index': tf.int32,
    'sentence': tfds.features.Text(),
    'highlight_indices': tfds.features.Text(),
})

# document_summary_law, document_summary_editorial, document_summary_newspaper
_DOCUMENT_QUALITY_SCORES = tfds.features.FeaturesDict({
    'readable': tf.int32,
    'accurate': tf.int32,
    'informative': tf.int32,
    'trustworthy': tf.int32,
})

_DOCUMENT_SUMMARY_LAW_FEATURE = tfds.features.FeaturesDict({ # document_summary_law
    'id': tfds.features.Text(),
    'category': tfds.features.Text(),
    'size': tfds.features.Text(),
    'char_count': tf.int32,
    'publish_date': tfds.features.Text(),
    'title': tfds.features.Text(),
    'text': _TEXT_FEATURE, 
    'annotator_id': tf.int32,
    'document_quality_scores': _DOCUMENT_QUALITY_SCORES,
    'extractive': tfds.features.Sequence(tf.int32), 
    'abstractive': tfds.features.Sequence(tfds.features.Text()),
})

# document_summary_editorial, document_summary_newspaper
_DOCUMENT_SUMMARY_FEATURE = tfds.features.FeaturesDict({ 
    'id': tfds.features.Text(),
    'category': tfds.features.Text(),
    'media_type': tfds.features.Text(),
    'media_sub_type': tfds.features.Text(),
    'media_name': tfds.features.Text(),
    'size': tfds.features.Text(),
    'char_count': tfds.features.Text(),
    'publish_date': tfds.features.Text(),
    'title': tfds.features.Text(),
    'text': _TEXT_FEATURE, 
    'annotator_id': tf.int32,
    'document_quality_scores': _DOCUMENT_QUALITY_SCORES,  
    'extractive': tfds.features.Sequence(tf.int32),
    'abstractive': tfds.features.Sequence(tfds.features.Text()),
})

_PERSONA_FEATURE = tfds.features.FeaturesDict({ # emotional_talk
    'persona-id': tfds.features.Text(),
    'human': tfds.features.Sequence(
        tfds.features.Text(),
    ),
    'computer': tfds.features.Sequence(
        tfds.features.Text(),
    ),
})

_EMOTION_FEATURE = tfds.features.FeaturesDict({ # emotional_talk
    'emotion-id': tfds.features.Text(),
    'type': tfds.features.Text(),
    'situation': tfds.features.Sequence(
        tfds.features.Text(),
    ),
})

_PROFILE_FEATURE = tfds.features.FeaturesDict({ # emotional_talk
    'persona-id': tfds.features.Text(),
    'persona': _PERSONA_FEATURE,
    'emotion': _EMOTION_FEATURE,
})

_CONTENT_FEATURE = tfds.features.FeaturesDict({ # emotional_talk
    'HS01': tfds.features.Text(),
    'SS01': tfds.features.Text(),
    'HS02': tfds.features.Text(),
    'SS02': tfds.features.Text(),
    'HS03': tfds.features.Text(),
    'SS03': tfds.features.Text(),
})

_TALK_FEATURE = tfds.features.FeaturesDict({ # emotional_talk
    'id': tfds.features.FeaturesDict({
        'profile-id': tfds.features.Text(),
        'talk-id': tfds.features.Text(),
    }),
    'content': _CONTENT_FEATURE,
})

_EMOTIONAL_TALK_FEATURE = tfds.features.FeaturesDict({ # emotional_talk
    'id': tf.int32, 
    'profile': _PROFILE_FEATURE,
    'talk': _TALK_FEATURE,
})

_DIALOG_INTENT_SEQUENCE = tfds.features.Sequence({ # dialog
    'a_entity': tfds.features.Text(),
    'a_morpheme': tfds.features.Text(),
    'answer': tfds.features.Text(),
    'q_entity': tfds.features.Text(),
    'q_morpheme': tfds.features.Text(),
    'question': tfds.features.Text(),
    'synonyms': tfds.features.Text(),
})

_SUB_INTENT_FEATURE = tfds.features.FeaturesDict({ # dialog
    'a_entity': tfds.features.Text(),
    'a_morpheme': tfds.features.Text(),
    'answer': tfds.features.Text(),
    'q_entity': tfds.features.Text(),
    'q_morpheme': tfds.features.Text(),
    'question': tfds.features.Text(),
    'synonyms': tfds.features.Text(),
})

_INTENT_FEATURE = tfds.features.FeaturesDict({ # dialog
    'intent': _DIALOG_INTENT_SEQUENCE,
    'main_intent': tfds.features.Text(),
    'sub_intent': tfds.features.Sequence({
        'intent': _SUB_INTENT_FEATURE,
        'sub_intent': tfds.features.Text(),
    })
})

_DIALOG_FEATURE = tfds.features.FeaturesDict({  # dialog
    'id': tf.int32, 
    'intent': _INTENT_FEATURE,
    'domain':tfds.features.Text(),
    'category': tfds.features.Text(),
})

_INTENT_SEQUENCE = tfds.features.Sequence({ # dialog_intent
    'SUB_INTENT': tfds.features.Sequence({
        'sub_intent': tfds.features.Text(),
    }),
    'intent': tfds.features.Text(),
})

_INTENT_FEATURE = tfds.features.FeaturesDict({ # dialog_intent
    'id': tf.int32,
    'intent': _INTENT_SEQUENCE,
    'domain': tfds.features.Text(),
    'category': tfds.features.Text(),
})

_HEADW0RD_SEQUENCE = tfds.features.Sequence({ # dialog_headword
    'WORD': tfds.features.Sequence({
        'word': tfds.features.Text(),
    }),
    'headword': tfds.features.Text(),
})

_HEADWORD_FEATURE = tfds.features.FeaturesDict({ # dialog_headword
    'id': tf.int32,
    'head': _HEADW0RD_SEQUENCE,
    'domain': tfds.features.Text(),
    'category': tfds.features.Text(),
})

_KNOWLEDGE_SUB_SEQUENCE = tfds.features.Sequence({ # dialog_knowledge
    'KNOWLEDGE': tfds.features.Sequence({
        'knowledge': tfds.features.Text(),
        'knowledge_detail': tfds.features.Text(),
    }),
    'sub_intent': tfds.features.Text(),
})

_KNOWLEDGE_SEQUENCE = tfds.features.Sequence({ # dialog_knowledge
    'SUB_INTENT': _KNOWLEDGE_SUB_SEQUENCE,
    'intent': tfds.features.Text(),
})

_KNOWLEDGE_FEATURE = tfds.features.FeaturesDict({ # dialog_knowledge
    'id': tf.int32,
    'knowledge': _KNOWLEDGE_SEQUENCE,
    'domain': tfds.features.Text(),
    'category': tfds.features.Text(),
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

def _parsing_dialog_intent(intent): # dialog
    keys = ['a_entity', 'a_morpheme', 'answer', 
                'q_entity', 'q_morpheme', 'question', 'synonyms']
    dic = {}
    for k in keys:
        try:
            dic[f'{k}'] = intent[f'{k}']
        except:
            dic[f'{k}'] = ''
    
    return dic

def _parsing_dialog_sub_intent(): # dialog
    dic = {}
    dic['INTENT'] = {} #_parsing_intent({})
    dic['SUB_INTENT'] = ''
    return [dic]

def _parsing_dialog(file_path): # dialog 
    with open(file_path, mode='r') as f:
        obj = json.loads(f.read())
        _id = 0

        for sample in obj['DATA']:
            _category = sample['CATEGORY']
            _domain = sample['DOMAIN']

            for c_lst in _category:
                intent_lst = c_lst['INTENT']
                category = c_lst['category']

                for intent in intent_lst: # intent_lst, main, sub_lst
                    
                    dic = {}
                    # intent
                    result_intent = []
                    for i in intent['INTENT']:
                        dic_intent = _parsing_dialog_intent(i)
                        result_intent.append(dic_intent)
                    dic['intent'] = result_intent

                    # main_intent
                    dic['main_intent'] = intent['MAIN_INTENT']

                    try:
                        sub_lst = intent['SUB_INTENT']
                    except KeyError: 
                        sub_lst = _parsing_dialog_sub_intent()
                    
                    result_sub_intent = []
                    
                    for sub_intent in sub_lst:
                        dic_sub_intent = {}
                        dic_sub_sub_intent = _parsing_dialog_intent(sub_intent)
                        dic_sub_intent['intent'] = dic_sub_sub_intent
                        dic_sub_intent['sub_intent'] = sub_intent['SUB_INTENT']
                        result_sub_intent.append(dic_sub_intent)

                    dic['sub_intent'] = result_sub_intent
                    
                    yield _id, {
                        'id': _id,
                        'intent': dic,
                        'domain': _domain,
                        'category': category,
                    }
                    _id += 1

def _parsing_empty_sub_intent(): # dialog/intent
    dic = {}
    dic['sub_intent'] = ""
    return [dic]

def _parsing_empty_intent(): # dialog/intent
    dic = {}
    dic['SUB_INTENT'] = _parsing_empty_sub_intent() 
    dic['intent'] = ""
    return [dic]

def _parsing_intent(file_path): # dialog/intent
    with open(file_path, mode='r') as f:
        obj = json.loads(f.read())
        _id = 0

        for sample in obj['DATA']:
            _category = sample['CATEGORY']
            _domain = sample['DOMAIN']

            for c_list in _category:
                category = c_list['category']
                intent_lst = c_list['INTENT']

                result = []
                if len(intent_lst) == 0:
                    result = _parsing_empty_intent()
                else:
                    for intent in intent_lst:
                        if len(intent['SUB_INTENT']) == 0:
                            intent['SUB_INTENT'] = _parsing_empty_sub_intent()
                        result.append(intent)

                yield _id, {
                    'id': _id,
                    'intent': result,
                    'domain': _domain,
                    'category': category,
                }
                _id += 1

def _parsing_empty_word():
    dic = {}
    dic['word'] = ""
    return [dic]

def _parsing_empty_head():
    dic = {}
    dic['WORD'] = _parsing_empty_word()
    dic['headword'] = ""
    return [dic]

def _parsing_headword(file_path):
    with open(file_path, mode='r') as f:
        obj = json.loads(f.read())
        _id = 0

        for sample in obj['DATA']:
            _category = sample['CATEGORY']
            _domain = sample['DOMAIN']

            for c_list in _category:
                category = c_list['category']
                head_lst = c_list['HEADWORD']

                result = []
                if len(head_lst) == 0:
                    result = _parsing_empty_head()
                else:
                    for head in head_lst:
                        if len(head['WORD']) == 0:
                            head['WORD'] = _parsing_empty_word()
                        result.append(head)

                yield _id, {
                    'id': _id,
                    'head': result,
                    'domain': _domain,
                    'category': category,
                }
                _id += 1

def _parsing_empty_sub_knowledge_lst(): # dialog/knowledge
    dic = {}
    dic['knowledge'] = ""
    dic['knowledge_detail'] = ""
    return [dic]

def _parsing_empty_sub_knowledge(): # dialog/knowledge
    dic = {}
    dic['KNOWLEDGE'] = _parsing_empty_sub_knowledge_lst()
    dic['sub_intent'] = ""
    return [dic]

def _parsing_empty_knowledge(): # dialog/knowledge
    dic = {}
    dic['SUB_INTENT'] = _parsing_empty_sub_knowledge() 
    dic['intent'] = ""
    return [dic]

def _parsing_knowledge(file_path): # dialog/knowledge
    with open(file_path, mode='r') as f:
        obj = json.loads(f.read())
        _id = 0

        for sample in obj['DATA']:
            _category = sample['CATEGORY']
            _domain = sample['DOMAIN']

            for c_list in _category:
                category = c_list['category']
                intent_lst = c_list['INTENT']

                result = []
                if len(intent_lst) == 0:
                    result = _parsing_empty_knowledge()
                else:
                    for intent in intent_lst:
                        if len(intent['SUB_INTENT']) == 0:
                            intent['SUB_INTENT'] = _parsing_empty_sub_knowledge()
                        else:
                            for sub_intent in intent['SUB_INTENT']:
                                if len(sub_intent['KNOWLEDGE']) == 0:
                                    sub_intent['KNOWLEDGE'] = _parsing_empty_sub_knowledge_lst()
                        result.append(intent)

                yield _id, {
                    'id': _id,
                    'knowledge': result,
                    'domain': _domain,
                    'category': category,
                }
                _id += 1

def _hash_text(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def _filter_fn_hash_id(uid, split_fn):
    hash_id = _hash_text(str(uid))
    val = int(hash_id, 16)
    return split_fn(val)

_DEFAULT_RAW_CORPUS_SPLIT = {
              'source': [tfds.Split.TRAIN],
              'split': {
                tfds.Split.TRAIN: lambda x: x % 1000 > 0,
                tfds.Split.VALIDATION: lambda x: x % 1000 == 0,
              }}

_DEFAULT_DOWNSTREAMTASK_CORPUS_SPLIT = {
              'source': [tfds.Split.TRAIN],
              'split': {
                tfds.Split.TRAIN: lambda x: x % 10 > 1,
                tfds.Split.VALIDATION: lambda x: x % 10 == 0,
                tfds.Split.TEST: lambda x: x % 10 == 1,
              }}

class AIHubConfig(tfds.core.BuilderConfig):
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

class AIHub(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for AIHub dataset."""

    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    BUILDER_CONFIGS = [
        AIHubConfig(
            name='common.squad.v1.0',
            data_root=_DATASET_ROOT['common_squad'],
            feature=_COMMON_SQUAD_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['nia_common_02_squad_질문, 답변, 제시문 말뭉치/ko_wiki_v1_squad.json']},
            reading_fn=_parsing_common_squad,
            parsing_fn=lambda x:x,
        ),
        AIHubConfig(
            name='common.squad.v1.0.split',
            data_root=_DATASET_ROOT['common_squad'],
            feature=_COMMON_SQUAD_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['nia_common_02_squad_질문, 답변, 제시문 말뭉치/ko_wiki_v1_squad.json']},
            reading_fn=_parsing_common_squad,
            parsing_fn=lambda x:x,
            split_fn=_DEFAULT_DOWNSTREAMTASK_CORPUS_SPLIT,
        ),
        AIHubConfig(
            name='paper.summary.v1.0.split',
            data_root=_DATASET_ROOT['paper_summary'],
            feature=_PAPER_SUMMARY_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['Training/training_논문/*.json'],
                          tfds.Split.VALIDATION: ['Validation/validation_논문/*.json']},
            reading_fn=_parsing_paper_summary,
            parsing_fn=lambda x:x,
        ),
        AIHubConfig(
            name='paper.patent.section.v1.0.split',
            data_root=_DATASET_ROOT['paper_patent_section'],
            feature=_PAPER_PATENT_SECTION_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['Training/training_특허섹션만/*.json'],
                          tfds.Split.VALIDATION: ['Validation/validation_특허섹션만/*.json']},
            reading_fn=_parsing_paper_patent_section,
            parsing_fn=lambda x:x,
        ),
        AIHubConfig(
            name='paper.patent.total.v1.0.split',
            data_root=_DATASET_ROOT['paper_patent_total'],
            feature=_PAPER_PATENT_TOTAL_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['Training/training_특허전체/*.json'],
                          tfds.Split.VALIDATION: ['Validation/validation_특허전체/*.json']},
            reading_fn=_parsing_paper_patent_total,
            parsing_fn=lambda x:x,
        ),
        AIHubConfig(
            name='document.summary.law.v1.0.split',
            data_root=_DATASET_ROOT['document_summary_law'],
            feature=_DOCUMENT_SUMMARY_LAW_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['1.Training/train_법률_data/법률문서/train_original.json'],
                          tfds.Split.VALIDATION: ['2.Validation/valid_법률_data/법률문서/dev_original.json']},
            reading_fn=_parsing_document_summary_law,
            parsing_fn=lambda x:x,
        ),
        AIHubConfig(
            name='document.summary.editorial.v1.0.split',
            data_root=_DATASET_ROOT['document_summary_editorial'],
            feature=_DOCUMENT_SUMMARY_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['1.Training/train_사설잡지_data/train_original.json'],
                          tfds.Split.VALIDATION: ['2.Validation/valid_사설잡지_data/dev_original.json']},
            reading_fn=_parsing_document_summary,
            parsing_fn=lambda x:x,
        ),
        AIHubConfig(
            name='emotional.talk.v1.0.split',
            data_root=_DATASET_ROOT['emotional_talk'],
            feature=_EMOTIONAL_TALK_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['Training/감성대화말뭉치(최종데이터)_Training/감성대화말뭉치(최종데이터)_Training.json'],
                          tfds.Split.VALIDATION: ['Validation/감성대화말뭉치(최종데이터)_Validation/감성대화말뭉치(최종데이터)_Validation.json']},
            reading_fn=_parsing_emotional_talk,
            parsing_fn=lambda x:x,
        ),
        AIHubConfig(
            name='dialog.v1.0',
            data_root=_DATASET_ROOT['dialog'],
            feature=_DIALOG_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['01_dialog/dialog/dialog.json']},
            reading_fn=_parsing_dialog,
            parsing_fn=lambda x:x,
        ),
        AIHubConfig(
            name='dialog.v1.0.split',
            data_root=_DATASET_ROOT['dialog'],
            feature=_DIALOG_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['01_dialog/dialog/dialog.json']},
            reading_fn=_parsing_dialog,
            parsing_fn=lambda x:x,
            split_fn=_DEFAULT_DOWNSTREAMTASK_CORPUS_SPLIT,
        ),
        AIHubConfig(
            name='dialog.intent.v1.0',
            data_root=_DATASET_ROOT['dialog_intent'],
            feature=_INTENT_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['02_intent/intent/intent.json']},
            reading_fn=_parsing_intent,
            parsing_fn=lambda x:x,
        ),
        AIHubConfig(
            name='dialog.intent.v1.0.split',
            data_root=_DATASET_ROOT['dialog_intent'],
            feature=_INTENT_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['02_intent/intent/intent.json']},
            reading_fn=_parsing_intent,
            parsing_fn=lambda x:x,
            split_fn=_DEFAULT_DOWNSTREAMTASK_CORPUS_SPLIT,
        ),
        AIHubConfig(
            name='dialog.headword.v1.0',
            data_root=_DATASET_ROOT['dialog_headword'],
            feature=_HEADWORD_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['03_headword/headword/headword.json']},
            reading_fn=_parsing_headword,
            parsing_fn=lambda x:x,
        ),
        AIHubConfig(
            name='dialog.headword.v1.0.split',
            data_root=_DATASET_ROOT['dialog_headword'],
            feature=_HEADWORD_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['03_headword/headword/headword.json']},
            reading_fn=_parsing_headword,
            parsing_fn=lambda x:x,
            split_fn=_DEFAULT_DOWNSTREAMTASK_CORPUS_SPLIT,
        ),
        AIHubConfig(
            name='dialog.knowledge.v1.0',
            data_root=_DATASET_ROOT['dialog_knowledge'],
            feature=_KNOWLEDGE_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['04_knowledge/knowledge/knowledge.json']},
            reading_fn=_parsing_knowledge,
            parsing_fn=lambda x:x,
        ),
        AIHubConfig(
            name='dialog.knowledge.v1.0.split',
            data_root=_DATASET_ROOT['dialog_knowledge'],
            feature=_KNOWLEDGE_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['04_knowledge/knowledge/knowledge.json']},
            reading_fn=_parsing_knowledge,
            parsing_fn=lambda x:x,
            split_fn=_DEFAULT_DOWNSTREAMTASK_CORPUS_SPLIT,
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
    DIALOG(v1.0): manual_dir/AIHub/dialog/01_dialog/dialog/dialog.json
    DIALOG_INTENT(v1.0): manual_dir/AIHub/dialog/02_intent/intent/intent.json
    DIALOG_HEADWORD(v1.0): manual_dir/AIHub/dialog/03_headword/headword/headword.json
    DIALOG_KNOWLEDGE(v1.0): manual_dir/AIHub/dialog/04_knowledge/knowledge/knowledge.json
    ============================================
    """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=self.builder_config.feature,
            homepage=self.builder_config.homepage,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path_kv = {}
        for k, v in self.builder_config.data_sp_path.items():
            path_list = []
            for vv in v:
                path_list.extend(glob.glob(os.path.join(
                    dl_manager.manual_dir, self.builder_config.data_root, vv)))
            path_kv[k] = path_list
            path_list = []

        for _, v in path_kv.items():
            if len(v) == 0:
                raise AssertionError("For the AIHub dataset, you must manually download and extract dataset under {0}/{1}.".format(
                    dl_manager.manual_dir,
                    self.builder_config.data_root
                ))

        if self.builder_config.split_fn is not None:
            in_files = []
            for sp_s_key in self.builder_config.split_fn['source']:
                in_files.extend(path_kv[sp_s_key])
            split_fn_kv = self.builder_config.split_fn['split']
            return {k:self._generate_examples(in_files, v) for k, v in split_fn_kv.items()}

        return {k:self._generate_examples(v) for k,v in path_kv.items()}

    def _generate_examples(self, path_list, split_fn=None):
        """Yields examples."""
        if split_fn is not None:
            split_filter = functools.partial(_filter_fn_hash_id, split_fn=split_fn)

        _hash_set = set()
        for file_path in path_list:
            try:
                for example in iter(self.builder_config.reading_fn(file_path)):
                    uid, ex = self.builder_config.parsing_fn(example)
                    
                    if split_fn is not None:
                        if not split_filter(str(uid)):
                            continue
                    hash_id = _hash_text(str(uid))
                    if hash_id not in _hash_set:
                        _hash_set.add(hash_id)
                        yield uid, ex
            except Exception as e:
                print(e)