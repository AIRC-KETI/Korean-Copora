"""klue dataset."""
import os
import csv
import json
import copy
import textwrap
import functools

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_datasets.public_api as tfds

# TODO(klue): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
# KLUE: Korean Language Understanding Evaluation 

The KLUE is introduced to make advances in Korean NLP. Korean pre-trained language models(PLMs) have appeared to solve Korean NLP problems since PLMs have brought significant performance gains in NLP problems in other languages. Despite the proliferation of Korean language models, however, none of the proper evaluation datasets has been opened yet. The lack of such benchmark dataset limits the fair comparison between the models and further progress on model architectures. 

Along with the benchmark tasks and data, we provide **suitable evaluation metrics** and fine-tuning recipes for pretrained language models for each task. We furthermore release the PLMs, **KLUE-BERT** and **KLUE-RoBERTa**, to help reproducing baseline models on KLUE and thereby facilitate future research. 

See [our paper](https://arxiv.org/pdf/2105.09680.pdf) for more details.


## Design Principles
In designing the Korean Language Understanding Evaluation (KLUE) benchmark, we aim to make KLUE; 

1. cover **diverse** tasks and corpora
2. **accessible** to everyone without any restriction
3. include **accurate** and unambiguous annotations
4. **mitigate** AI ethical issues. 


## Benchmark Datasets
KLUE benchmark is composed of 8 tasks:
- Topic Classification (TC)
- Sentence Textual Similarity (STS)
- Natural Language Inference (NLI)
- Named Entity Recognition (NER)
- Relation Extraction (RE)
- (Part-Of-Speech) + Dependency Parsing (DP)
- Machine Reading Comprehension (MRC)
- Dialogue State Tracking (DST)

`NOTE`: In the paper, we describe more in detail how our 4 principles have guided creating KLUE from task selection, corpus selection, annotation protocols, determining evaluation metrics to baseline construction. 
"""

# TODO(klue): BibTeX citation
_CITATION = """
@misc{park2021klue,
      title={KLUE: Korean Language Understanding Evaluation}, 
      author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jungwoo Ha and Kyunghyun Cho Alice Oh Jungwoo Ha Kyunghyun Cho},
      year={2021},
      eprint={2105.09680},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

# version of the klue dataset
_VERSION = tfds.core.Version('1.0.0')

_RELEASE_NOTES = {
    '1.0.0': 'Initial release.',
}

_HOME_PAGE = 'https://klue-benchmark.com/'


_KLUE_URL = 'https://github.com/KLUE-benchmark/KLUE'
_KLUE_ROOT = 'https://raw.githubusercontent.com/KLUE-benchmark/KLUE/main/klue_benchmark/'

# --------------------------------------------------------------------------
# TC: Topic classification - Yonhap News Agency Topic Classification(YNAT)
# --------------------------------------------------------------------------
_KLUE_TC_TRAIN_LINK = os.path.join(_KLUE_ROOT, 'ynat-v1/ynat-v1_train.json')
_KLUE_TC_DEV_LINK = os.path.join(_KLUE_ROOT, 'ynat-v1/ynat-v1_dev.json')
_KLUE_TC_DATA_URL = {
    'train': _KLUE_TC_TRAIN_LINK, 'dev': _KLUE_TC_DEV_LINK}

_KLUE_TC_DESCRIPTION = textwrap.dedent("""\
            In topic classification (TC), the goal is to predict the topic of a given 
            text snippet. We include TC in our KLUE benchmark, as inferring the topic 
            of a text is a key capability that should be possessed by a language 
            understanding system. Following a typical single sentence classification 
            task, we introduce YNAT, a Younhap News Agency news headlines for Topic 
            Classification. For Korean, no dataset has been proposed for this task, 
            which motivates us to construct the first Korean topic classification 
            benchmark.
            
            In this task, given a news headline, a text classifier must predict a 
            topic which is one of politics, economy, society, culture, world, IT/science, 
            and sports. Macro-F1 score is used to evaluate a system.""")

_KLUE_TC_CLASSES = [
    '정치',  # politics
    '경제',  # economy
    '사회',  # society
    '생활문화',  # culture
    '세계',  # world
    'IT과학',  # IT/science
    '스포츠',  # sports
    '해당없음'  # OOD(out-of-distribution)
]
_KLUE_TC_LABEL_FEATURE = tfds.features.ClassLabel(names=_KLUE_TC_CLASSES)
_KLUE_TC_FULL_FEATURES = tfds.features.FeaturesDict({
    "guid":
        tfds.features.Text(),
    "title":
        tfds.features.Text(),
    "predefined_news_category":
        _KLUE_TC_LABEL_FEATURE,
    "label":
        _KLUE_TC_LABEL_FEATURE,
    "url":
        tfds.features.Text(),
    "date":
        tfds.features.Text(),
    "annotations": tfds.features.FeaturesDict({
        "annotators": tfds.features.Sequence(tf.int32),
        "annotations": tfds.features.FeaturesDict({
            "first-scope": tfds.features.Sequence(tfds.features.Text()),
            "second-scope": tfds.features.Sequence(tfds.features.Text()),
            "third-scope": tfds.features.Sequence(tfds.features.Text()),
        })
    })
})

_KLUE_TC_FEATURES = tfds.features.FeaturesDict({
    "guid":
        tfds.features.Text(),
    "title":
        tfds.features.Text(),
    "label":
        _KLUE_TC_LABEL_FEATURE
})
_KLUE_TC_FEATURE_KEYS = ["guid", "title", "label"]

# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# STS: Semantic Textual Similarity
# --------------------------------------------------------------------------
_KLUE_STS_TRAIN_LINK = os.path.join(
    _KLUE_ROOT, 'klue-sts-v1/klue-sts-v1_train.json')
_KLUE_STS_DEV_LINK = os.path.join(
    _KLUE_ROOT, 'klue-sts-v1/klue-sts-v1_dev.json')
_KLUE_STS_DATA_URL = {
    'train': _KLUE_STS_TRAIN_LINK, 'dev': _KLUE_STS_DEV_LINK}

_KLUE_STS_DESCRIPTION = textwrap.dedent("""\
            Semantic Textual Similarity (STS) is to measure the degree of 
            semantic equivalence between two sentences. We include KLUE-STS 
            in our benchmark because it is essential to other NLP tasks such 
            as machine translation, summarization, and question answering. 
            Like STS in GLUE, many NLU benchmarks include comparing semantic 
            similarity of text snippets such as semantic similarity, 
            paraphrase detection, or word sense disambiguation.
            
            We formulate STS as a sentence pair regression task which 
            predicts the semantic similarity of two input sentences as a 
            real value from 0 (no meaning overlap) to 5 (meaning equivalence). 
            A model performance is measured by Pearson's correlation coefficient. 
            We additionally binarize the real numbers into two classes with 
            a threshold score 3.0 (paraphrased or not), and use F1 score to 
            evaluate the model.""")

_KLUE_STS_FULL_FEATURES = tfds.features.FeaturesDict({
    "guid":
        tfds.features.Text(),
    "source":
        tfds.features.Text(),
    "sentence1":
        tfds.features.Text(),
    "sentence2":
        tfds.features.Text(),
    "label":
        tf.float32,
    "labels":
        tfds.features.FeaturesDict({
            "label": tf.float32,
            "real-label": tf.float32,
            "binary-label": tf.int32
        }),
    "annotations": tfds.features.FeaturesDict({
        "agreement": tfds.features.Text(),
        "annotators": tfds.features.Sequence(tfds.features.Text()),
        "annotations": tfds.features.Sequence(tf.int32),
    })
})

_KLUE_STS_FEATURES = tfds.features.FeaturesDict({
    "guid":
        tfds.features.Text(),
    "sentence1":
        tfds.features.Text(),
    "sentence2":
        tfds.features.Text(),
    "label":
        tf.float32,
})
_KLUE_STS_FEATURE_KEYS = ["guid", "sentence1", "sentence2", "label"]
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# NLI: Natural Language Inference
# --------------------------------------------------------------------------
_KLUE_NLI_TRAIN_LINK = os.path.join(
    _KLUE_ROOT, 'klue-nli-v1/klue-nli-v1_train.json')
_KLUE_NLI_DEV_LINK = os.path.join(
    _KLUE_ROOT, 'klue-nli-v1/klue-nli-v1_dev.json')
_KLUE_NLI_DATA_URL = {
    'train': _KLUE_NLI_TRAIN_LINK, 'dev': _KLUE_NLI_DEV_LINK}

_KLUE_NLI_DESCRIPTION = textwrap.dedent("""\
            The goal of Natural language inference (NLI) is to infer the relationship 
            between the hypothesis sentence and the premise sentence. Given a premise, 
            an NLI model determines if hypothesis is true (entailment), false 
            (contradiction), or undetermined (neutral). The task is also known as 
            Recognizing Textual Entailment (RTE). We include KLUE-NLI since 
            understanding entailment and contradiction between sentences is fundamental 
            to NLU. NLI datasets are also included in various NLU benchmarks such as 
            GLUE and superGLUE, and they are valuable as training data for other 
            NLU tasks.
            
            We formulate NLI as a sentence pair classification task where an NLI model 
            reads each pair of premise and hypothesis sentences and predicts whether 
            the relationship is entailment, contradiction, or neutral. We use the 
            classification accuracy to measure the model performance.""")

_KLUE_NLI_CLASSES = [
    'entailment',
    'contradiction',
    'neutral'
]
_KLUE_NLI_LABEL_FEATURE = tfds.features.ClassLabel(names=_KLUE_NLI_CLASSES)

_KLUE_NLI_FULL_FEATURES = tfds.features.FeaturesDict({
    "guid":
        tfds.features.Text(),
    "source":
        tfds.features.Text(),
    "premise":
        tfds.features.Text(),
    "hypothesis":
        tfds.features.Text(),
    "gold_label":
        _KLUE_NLI_LABEL_FEATURE,
    "author":
        _KLUE_NLI_LABEL_FEATURE,
    "label2":
        _KLUE_NLI_LABEL_FEATURE,
    "label3":
        _KLUE_NLI_LABEL_FEATURE,
    "label4":
        _KLUE_NLI_LABEL_FEATURE,
    "label5":
        _KLUE_NLI_LABEL_FEATURE,
})

_KLUE_NLI_FEATURES = tfds.features.FeaturesDict({
    "guid":
        tfds.features.Text(),
    "premise":
        tfds.features.Text(),
    "hypothesis":
        tfds.features.Text(),
    "gold_label":
        _KLUE_NLI_LABEL_FEATURE,
})
_KLUE_NLI_FEATURE_KEYS = ["guid", "premise", "hypothesis", "gold_label"]

# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# NER: Named Entity Recognition
# --------------------------------------------------------------------------

_KLUE_NER_TRAIN_LINK = os.path.join(
    _KLUE_ROOT, 'klue-ner-v1/klue-ner-v1_train.tsv')
_KLUE_NER_DEV_LINK = os.path.join(
    _KLUE_ROOT, 'klue-ner-v1/klue-ner-v1_dev.tsv')
_KLUE_NER_DATA_URL = {
    'train': _KLUE_NER_TRAIN_LINK, 'dev': _KLUE_NER_DEV_LINK}

_KLUE_NER_DESCRIPTION = textwrap.dedent("""\
            The goal of named entity recognition (NER) is to detect the boundaries 
            of named entities in unstructured text and to classify the types. A 
            named entity can be of one of predefined entity types such as person, 
            location, organization, time expressions, quantities and monetary values. 
            KLUE-NER is included in KLUE, since recognizing entities is essential 
            to build a system for extracting knowledge from unstructured texts. 
            Due to its importance, other NLU benchmarks also contain NER datasets. 
            Although there exist a few NER resources in Korean, they do not cover 
            web texts. This leads to underperforming of NER models in a web domain. 
            This issue is the motivation to create a fully accessible Korean NER 
            dataset that encompases multiple domains including web texts.
            
            A model should detect the spans and classify the types of entities 
            included in an input sentence. The six entity types used in KLUE-NER 
            are person(PS), location(LC), organization(OG), date(DT), time(TI), 
            and quantity(QT). They are tagged via character-level BIO 
            (Begin-Inside-Outside) tagging scheme, and thus we evaluate a model’s 
            performance using entity-level macro F1 and character-level macro F1 
            scores. Our F1 score weights recall and precision equally to incentivize 
            a model which maximize both precision and recall simultaneously.""")

_KLUE_NER_TAGS = [
    'PS',  # person
    'LC',  # location
    'OG',  # organization
    'DT',  # date
    'TI',  # time
    'QT'  # quantity
]

_KLUE_NER_IOB2_TAGS = [
    'O',
    'B-PS',
    'I-PS',
    'B-LC',
    'I-LC',
    'B-OG',
    'I-OG',
    'B-DT',
    'I-DT',
    'B-TI',
    'I-TI',
    'B-QT',
    'I-QT',
]

_KLUE_NER_LABEL_FEATURE = tfds.features.ClassLabel(names=_KLUE_NER_TAGS)
_KLUE_NER_IOB2_LABEL_FEATURE = tfds.features.ClassLabel(
    names=_KLUE_NER_IOB2_TAGS)

_KLUE_NE_LABEL_SEQ_FEATURE = tfds.features.Sequence({
    "form": tfds.features.Text(),
    "begin": tf.int32,
    "end": tf.int32,
    "label": _KLUE_NER_LABEL_FEATURE,
})

# conll shared task format... and character level...
_KLUE_NER_FULL_FEATURES = tfds.features.FeaturesDict({
    "guid":
        tfds.features.Text(),
    "char":
        tfds.features.Sequence(tfds.features.Text()),
    "ne_tag":
        tfds.features.Sequence(_KLUE_NER_IOB2_LABEL_FEATURE),
    "text":
        tfds.features.Text(),
    "NE": _KLUE_NE_LABEL_SEQ_FEATURE,
})

# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# RE: Relation Extraction
# --------------------------------------------------------------------------
_KLUE_RE_TRAIN_LINK = os.path.join(
    _KLUE_ROOT, 'klue-re-v1/klue-re-v1_train.json')
_KLUE_RE_DEV_LINK = os.path.join(_KLUE_ROOT, 'klue-re-v1/klue-re-v1_dev.json')
_KLUE_RE_RELATION_LINK = os.path.join(
    _KLUE_ROOT, 'klue-re-v1/relation_list.json')
_KLUE_RE_DATA_URL = {
    'train': _KLUE_RE_TRAIN_LINK, 'dev': _KLUE_RE_DEV_LINK, 'relations': _KLUE_RE_RELATION_LINK}

_KLUE_RE_DESCRIPTION = textwrap.dedent("""\
            Relation extraction (RE) identifies semantic relations between entity 
            pairs in a text. The relation is defined between an entity pair consisting 
            of subject entity and object entity. For example, in a sentence 
            `Kierkegaard was born to an affluent family in Copenhagen’, the subject 
            entity is `Kierkegaard’ and the object entity is `Copenhagen’. The goal is 
            then to pick an appropriate relationship between these two entities: 
            place_of_birth. In order to evaluate whether a model correctly understands the 
            relationships between entities, we include KLUE-RE in our benchmark. 
            Since there is no large-scale RE benchmark publicly available in Korean, 
            we collect and annotate our own dataset.
            
            We formulate RE as a single sentence classification task. A model picks 
            one of predefined relation types describing the relation between two 
            entities within a given sentence. In other words, an RE model predicts
            an appropriate relation r of entity pair (e_subj, e_obj) in a sentence s, 
            where e_subj is the subject entity and e_obj is the object entity. We refer 
            to (e_subj, r, e_obj) as a relation triplet. The entities are marked as 
            corresponding spans in each sentence s. There are 30 relation classes that 
            consist of 18 person-related relations, 11 organization-related relations, 
            and no_relation. We evaluate a model using micro-F1 score, computed after 
            excluding no_relation, and area under the precision-recall curve (AUPRC) 
            including all 30 classes.""")

_KLUE_RE_RELATIONS = [
    "no_relation",
    "org:dissolved",
    "org:founded",
    "org:place_of_headquarters",
    "org:alternate_names",
    "org:member_of",
    "org:members",
    "org:political/religious_affiliation",
    "org:product",
    "org:founded_by",
    "org:top_members/employees",
    "org:number_of_employees/members",
    "per:date_of_birth",
    "per:date_of_death",
    "per:place_of_birth",
    "per:place_of_death",
    "per:place_of_residence",
    "per:origin",
    "per:employee_of",
    "per:schools_attended",
    "per:alternate_names",
    "per:parents",
    "per:children",
    "per:siblings",
    "per:spouse",
    "per:other_family",
    "per:colleagues",
    "per:product",
    "per:religion",
    "per:title"
]
_KLUE_RE_LABEL_FEATURE = tfds.features.ClassLabel(names=_KLUE_RE_RELATIONS)

_KLUE_RE_ENTITY_TYPE = [
    "PER",
    "ORG",
    "POH",
    "DAT",
    "LOC",
    "NOH"
]
_KLUE_RE_ENTITY_TYPE_FEATURE = tfds.features.ClassLabel(
    names=_KLUE_RE_ENTITY_TYPE)

_KLUE_RE_ENTITY_FEATURE = tfds.features.FeaturesDict({
    "word": tfds.features.Text(),
    "start_idx": tf.int32,
    "end_idx": tf.int32,
    "type": _KLUE_RE_ENTITY_TYPE_FEATURE
})

_KLUE_RE_FULL_FEATURES = tfds.features.FeaturesDict({
    "guid":
        tfds.features.Text(),
    "sentence":
        tfds.features.Text(),
    "subject_entity":
        _KLUE_RE_ENTITY_FEATURE,
    "object_entity":
        _KLUE_RE_ENTITY_FEATURE,
    "label": _KLUE_RE_LABEL_FEATURE,
    "source":
        tfds.features.Text(),
})


# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# DP: Dependency Parsing
# --------------------------------------------------------------------------
_KLUE_DP_TRAIN_LINK = os.path.join(
    _KLUE_ROOT, 'klue-dp-v1/klue-dp-v1_train.tsv')
_KLUE_DP_DEV_LINK = os.path.join(_KLUE_ROOT, 'klue-dp-v1/klue-dp-v1_dev.tsv')
_KLUE_DP_DATA_URL = {
    'train': _KLUE_DP_TRAIN_LINK, 'dev': _KLUE_DP_DEV_LINK}

_KLUE_DP_DESCRIPTION = textwrap.dedent("""\
            Dependency Parsing (DP) is an NLP task that aims at finding relational 
            information among words. It has been an important component in many NLP 
            systems, because of its ability to capture the syntactic feature of a 
            sentence. We include KLUE-DP in the benchmark to evaluate the 
            representational power of language models in terms of syntactic features. 
            Formally, a dependency parser predicts a graph structure of an input 
            sentence based on the dependency grammar. In general, a parsed tree 
            consists of dependency arcs, connecting dependents to their heads, and 
            the dependency labels attached to the arcs that define the relations 
            between dependents and their heads. For example, the below figure shows 
            a parsed result of the example sentence: "철수가 사과를 먹었다 (Chul-Soo ate 
            an apple.)". In the tree, arrows depart from head and point to their 
            dependents. Thus '철수가 (Chul-Soo)' and '사과를 (an apple)' are dependents 
            of '먹었다(ate)' and '먹었다(ate)' is the head of '철수가 (Chul-Soo)' and 
            '사과를 (an apple)'. Also, '철수가 (Chul-Soo)' is dependent on '먹었다 (ate)' 
            with a "Subject" relation. We call this relation types as dependency 
            relation label DEPREL. For DEPREL, we follow the TTA Dependency 
            annotation (https://aiopen.etri.re.kr/data/003.%EC%9D%98%EC%A1%B4%EA%B5%AC%EB%AC%B8%EB%B6%84%EC%84%9D_%EA%B0%80%EC%9D%B4%EB%93%9C%EB%9D%BC%EC%9D%B8.pdf) 
            scheme consisting of a combination of 9 syntax tags and 6 function tags.
            
            Since each word in a sentence has a pair of dependency information 
            (Head, DEPREL), DP is conventionally formulated as a word-level sequence 
            tagging task. We evaluate a model’s performance using a unlabeled 
            attachment score (UAS) and labeled attachment score (LAS). During the 
            evaluation, labels with a cumulative frequency of 1\% from the bottom 
            are grouped into the OTHERS label to compensate for the negative impact 
            of lower frequency labels on LAS.""")

_KLUE_DP_SYNTAX = [
    "NP",  # Noun Phrase
    "VP",  # Verb Phrase
    "AP",  # Adverb Phrase
    "VNP",  # Copula Phrase
    "DP",  # Adnoun Phrase
    "IP",  # Interjection Phrase
    "X",  # Pseudo Phrase
    "L",  # Left Parenthesis and Quotation Mark
    "R"  # Right Parenthesis and Quotation Mark
]
_KLUE_DP_FUNC = [
    "SBJ",  # Subject
    "OBJ",  # Object
    "MOD",  # Noun Modifier
    "AJT",  # Predicate Modifier
    "CMP",  # Complement
    "CNJ",  # Conjunction
]

_KLUE_DP_DEPREL_TAGS = [
    "NP",
    "NP_AJT",
    "VP",
    "NP_SBJ",
    "VP_MOD",
    "NP_OBJ",
    "AP",
    "NP_CNJ",
    "NP_MOD",
    "VNP",
    "DP",
    "VP_AJT",
    "VNP_MOD",
    "NP_CMP",
    "VP_SBJ",
    "VP_CMP",
    "VP_OBJ",
    "VNP_CMP",
    "AP_MOD",
    "X_AJT",
    "VNP_AJT",
    "VP_CNJ",
    "IP",
    "X",
    "VNP_OBJ",
    "X_SBJ",
    "X_OBJ",
    "VNP_SBJ",
    "L",
    "AP_AJT",
    "X_CMP",
    "X_CNJ",
    "X_MOD",
    "AP_CMP",
    "R",
    "VNP_CNJ",
    "AP_SBJ",
    "NP_SVJ"
]
_KLUE_DP_DEPREL_TYPE_FEATURE = tfds.features.ClassLabel(
    names=_KLUE_DP_DEPREL_TAGS)

_KLUE_DP_SEQ_FEATURE = tfds.features.Sequence({
    "word_id": tf.int32,
    "word_form": tfds.features.Text(),
    "lemma": tfds.features.Text(),
    "POS": tfds.features.Text(),
    "head": tf.int32,
    "label": _KLUE_DP_DEPREL_TYPE_FEATURE,
})

_KLUE_DP_FULL_FEATURES = tfds.features.FeaturesDict({
    "guid":
        tfds.features.Text(),
    "form":
        tfds.features.Text(),
    "DP": _KLUE_DP_SEQ_FEATURE,
})


# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# MRC: Machine Reading Comprehension
# --------------------------------------------------------------------------
_KLUE_MRC_TRAIN_LINK = os.path.join(
    _KLUE_ROOT, 'klue-mrc-v1/klue-mrc-v1_train.json')
_KLUE_MRC_DEV_LINK = os.path.join(
    _KLUE_ROOT, 'klue-mrc-v1/klue-mrc-v1_dev.json')
_KLUE_MRC_DATA_URL = {
    'train': _KLUE_MRC_TRAIN_LINK, 'dev': _KLUE_MRC_DEV_LINK}

_KLUE_MRC_DESCRIPTION = textwrap.dedent("""\
            MRC is a task of evaluating model that can answer a question about 
            a given text passage. We formulate KLUE-MRC as to predict the answer 
            span in the given passage corresponding to the question. The input is 
            a concatenated sequence of the question and the passage separated with 
            a delimiter. The output is the start and end positions of the predicted 
            answer span within the passage.
            
            We provide three question types: paraphrase, multi-sentence reasoning, 
            and unanswerable, in order to evaluate different aspects of machine 
            reading capability of a model. These question types prevent a model 
            from exploiting reasoning shortcuts with simple word-matching by 
            enforcing lexical and syntactic variations when workers generate 
            questions. The questions also should be answered by considering the 
            full query sentence.""")

_KLUE_MRC_FEATURES = tfds.features.FeaturesDict({
    "guid":
        tf.string,
    "title":
        tfds.features.Text(),
    "context":
        tfds.features.Text(),
    "plausible_answers":
        tfds.features.Sequence({
            "text": tfds.features.Text(),
            "answer_start": tf.int32,
        }),
    "question":
        tfds.features.Text(),
    "is_impossible":
        tf.bool,
    "answers":
        tfds.features.Sequence({
            "text": tfds.features.Text(),
            "answer_start": tf.int32,
        }),
    "question_type": tf.int32,
    "source": tfds.features.Text(),
    "news_category": tfds.features.Text(),
})
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# DST: Dialogue State Tracking - Wizard of Seoul (WoS)
# --------------------------------------------------------------------------

_KLUE_DST_TRAIN_LINK = os.path.join(
    _KLUE_ROOT, 'wos-v1/wos-v1_train.json')
_KLUE_DST_DEV_LINK = os.path.join(
    _KLUE_ROOT, 'wos-v1/wos-v1_dev.json')
_KLUE_DST_ONTOLOGY_LINK = os.path.join(
    _KLUE_ROOT, 'wos-v1/ontology.json')
_KLUE_DST_DATA_URL = {
    'train': _KLUE_DST_TRAIN_LINK, 'dev': _KLUE_DST_DEV_LINK, 'ontology': _KLUE_DST_ONTOLOGY_LINK}

_KLUE_DST_DESCRIPTION = textwrap.dedent("""\
            Building a human-computer conversation system has been increasingly 
            attracting attention, and a task-oriented dialogue system is one 
            type of the dialogue systems. Dialogue State Tracking (DST) is 
            inferring dialogue states of an agent from task-oriented dialogue 
            system. As illustrated in the below figure, dialogue states are sets 
            of slot and value pairs that are relevant categories (e.g. hotel type) 
            and their possible values (e.g. guest house, hotel, motel), 
            respectively. Such task-oriented dialogue dataset is involved into 
            decaNLP benchmark tasks and dialoGLUE, the first task-oriented 
            dialogue benchmarks. In line with the growing attention on dialogue 
            tasks, we include DST in KLUE benchmarks.
            
            A dialogue state tracking system predicts slot and value pairs after 
            each user's utterance, and the potential pairs are predefined by task 
            schema and knowledge base (KB), tied to the choice of a scenario. 
            To evaluate the system, we use joint goal accuracy (JGA) and slot F1 score 
            (Slot F1). The JGA checks if all of the predicted slot-value pairs 
            are exactly matched with the ground-truths at every turn, and the 
            Slot F1 computes f1 score for each slot-value pair.""")

_KLUE_DST_FEATURES = tfds.features.FeaturesDict({
    "guid":
        tf.string,
    "domains": tfds.features.Sequence(tfds.features.Text()),
    "dialogue": tfds.features.Sequence({
      "role": tfds.features.Text(),
      "text": tfds.features.Text(),
      "state": tfds.features.Sequence(tfds.features.Text()),
    })
})

_KLUE_DST_GEN_FEATURES = tfds.features.FeaturesDict({
    "guid":
        tf.string,
    "domains": tfds.features.Sequence(tfds.features.Text()),
    "dialogue": tfds.features.Sequence({
      "role": tfds.features.Text(),
      "text": tfds.features.Text(),
      "state": tfds.features.Sequence({
        'domain-slot': tfds.features.Text(),
        'value': tfds.features.Text()
      }),
    })
})

# --------------------------------------------------------------------------


def parsing_json_examples_basic(filepath):
    with tf.io.gfile.GFile(filepath) as f:
        examples = json.load(f)
        for example in examples:
            yield example['guid'], example


def parsing_ner_examples(filepath):
    with tf.io.gfile.GFile(filepath) as f:
        comment = ''
        chrs = []
        tags = []
        for line in f.readlines():
            row = line.split('\t')
            if row[0].startswith('#'):
                comment = row[0]
            elif row[0] == '\n':
                guid = comment.split(' ')[-1]
                text, ne_seq = create_ner_example(chrs, tags)
                yield guid, {
                    'guid': guid,
                    'char': chrs,
                    'ne_tag': tags,
                    'text': text,
                    'NE': ne_seq
                }
                chrs.clear()
                tags.clear()
            else:
                chrs.append(row[0])
                tags.append(row[1].rstrip())

        if len(chrs) > 0:
            guid = comment.split(' ')[-1]
            text, ne_seq = create_ner_example(chrs, tags)
            yield guid, {
                'guid': guid,
                'char': chrs,
                'ne_tag': tags,
                'text': text,
                'NE': ne_seq
            }
            chrs.clear()
            tags.clear()


def create_ner_example(chrs, tags):
    text = ''.join(chrs)
    ne_seq = []

    start_idx = 0
    tag_stack = []
    chr_stack = []
    for t_idx, tag in enumerate(tags):
        if tag.startswith('B'):
            if len(chr_stack) > 0:
                form = ''.join(chr_stack)
                ne_seq.append({'form': form, 'begin': start_idx,
                              'end': start_idx+len(form), 'label': tag_stack[0]})
                chr_stack.clear()
                tag_stack.clear()

            start_idx = t_idx
            tag_stack.append(tag.split('-')[-1])
            chr_stack.append(chrs[t_idx])
        elif tag.startswith('I'):
            chr_stack.append(chrs[t_idx])
        else:
            if len(chr_stack) > 0:
                form = ''.join(chr_stack)
                ne_seq.append({'form': form, 'begin': start_idx,
                              'end': start_idx+len(form), 'label': tag_stack[0]})
                chr_stack.clear()
                tag_stack.clear()
    return text, ne_seq


def parsing_dp_examples(filepath):
    with tf.io.gfile.GFile(filepath) as f:
        comment = ''

        # tuple (index, word_form, lemma, pos, head, deprel)
        items = []

        for line in f.readlines():
            row = line.split('\t')
            if row[0].startswith('#'):
                comment = row[0]
            elif row[0] == '\n':
                guid = comment.split(' ')[-1]
                text, dp_seq = create_dp_example(items)
                yield guid, {
                    'guid': guid,
                    'form': text,
                    'DP': dp_seq
                }
                items.clear()
            else:
                row[-1] = row[-1].rstrip()
                items.append(row)

        if len(items) > 0:
            guid = comment.split(' ')[-1]
            text, dp_seq = create_dp_example(items)
            yield guid, {
                'guid': guid,
                'form': text,
                'DP': dp_seq
            }
            items.clear()


def create_dp_example(items):
    text = ' '.join([item[1] for item in items])
    dp_seq = []
    for item in items:
        dp_seq.append({
            "word_id": item[0],
            "word_form": item[1],
            "lemma": item[2],
            "POS": item[3],
            "head": item[4],
            "label": item[5],
        })
    return text, dp_seq


def parsing_mrc_examples(filepath):
    with tf.io.gfile.GFile(filepath) as f:
        klue_mrc = json.load(f)
        for article in klue_mrc["data"]:
            title = article.get("title", "")
            source = article.get("source", "")
            news_category = article.get("news_category", "")
            if news_category is None:
                news_category = ""
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    id_ = qa["guid"]

                    #  Not all examples have plausible answers
                    if "plausible_answers" not in qa:
                        qa["plausible_answers"] = []

                    question = qa["question"]
                    is_impossible = qa["is_impossible"]

                    plausible_answer_starts = [
                        plausible_answer["answer_start"]
                        for plausible_answer in qa["plausible_answers"]
                    ]
                    plausible_answers = [
                        plausible_answer["text"]
                        for plausible_answer in qa["plausible_answers"]
                    ]

                    answer_starts = [answer["answer_start"]
                                     for answer in qa["answers"]]
                    answers = [answer["text"] for answer in qa["answers"]]

                    question_type = qa.get("question_type", -1)

                    yield id_, {
                        "title": title,
                        "context": context,
                        "question": question,
                        "guid": id_,
                        "plausible_answers": {
                            "answer_start": plausible_answer_starts,
                            "text": plausible_answers,
                        },
                        "answers": {
                            "answer_start": answer_starts,
                            "text": answers,
                        },
                        "is_impossible": is_impossible,
                        "question_type": question_type,
                        "source": source,
                        "news_category": news_category
                    }

def parsing_dst_examples(filepath):
    with tf.io.gfile.GFile(filepath) as f:
        klue_wos = json.load(f)
        for example in klue_wos:
            guid = example.get("guid", "")
            domains = example.get("domains", [])
            dialogue_fmt = []
            dialogue = example.get("dialogue", [])
            for utterance in dialogue:
                role = utterance["role"]
                text = utterance["text"]
                state = utterance.get("state", [])
                dialogue_fmt.append({
                  "role": role,
                  "text": text,
                  "state": state
                })
            yield guid, {
              "guid": guid,
              "domains": domains,
              "dialogue": dialogue_fmt
            }

def parsing_dst_gen_examples(filepath, ontology_path):
    with tf.io.gfile.GFile(ontology_path) as f:
        ontology = json.load(f)
    with tf.io.gfile.GFile(filepath) as f:
        klue_wos = json.load(f)
        for example in klue_wos:
            guid = example.get("guid", "")
            domains = example.get("domains", [])
            dialogue_fmt = []
            dialogue = example.get("dialogue", [])
            for utterance in dialogue:
                role = utterance["role"]
                text = utterance["text"]
                state = utterance.get("state", [])
                state_dict = {'-'.join(x.split('-')[:-1]):x.split('-')[-1] for x in state}
                state_list = [{'domain-slot': k, 'value': state_dict[k]} if k in state_dict else {'domain-slot': k, 'value': 'none'} for k, v in ontology.items()]
                dialogue_fmt.append({
                  "role": role,
                  "text": text,
                  "state": state_list
                })
            yield guid, {
              "guid": guid,
              "domains": domains,
              "dialogue": dialogue_fmt
            }



def reduce_features(example, feature_keys):
    _uid, _example = example
    return _uid, {k: _example[k] for k in feature_keys}


def sts_cp_label(example):
    _uid, _example = example
    _example['label'] = _example['labels']['label']
    return _uid, _example


def nli_ch_key(example):
    _uid, _example = example
    if 'source' not in _example and 'genre' in _example:
        _example['source'] = _example['genre']
        del _example['genre']
    return _uid, _example


def chain(*funcs):
    def chained_call(arg):
        return functools.reduce(lambda r, f: f(r), funcs, arg)
    return chained_call


def re_incr_entity_end_idx(example):
    _uid, _example = example
    _example['subject_entity']['end_idx'] += 1
    _example['object_entity']['end_idx'] += 1
    return _uid, _example


class KlueConfig(tfds.core.BuilderConfig):
    def __init__(self,
                 name,
                 features,
                 data_url,
                 description="",
                 citation="",
                 parsing_fn=None,
                 process_fn=lambda x: x,
                 **kwargs):
        super(KlueConfig, self).__init__(
            name=name,
            version=_VERSION,
            release_notes=_RELEASE_NOTES,
            **kwargs
        )
        self.features = features
        self.data_url = data_url
        self.description = description
        self.citation = citation
        self.parsing_fn = parsing_fn
        self.process_fn = process_fn


class Klue(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for klue dataset."""

    BUILDER_CONFIGS = [
        KlueConfig(
            name='tc',
            features=_KLUE_TC_FEATURES,
            data_url=_KLUE_TC_DATA_URL,
            description=_KLUE_TC_DESCRIPTION,
            citation="",  # The KLUE citation is sufficient.
            parsing_fn=parsing_json_examples_basic,
            process_fn=functools.partial(
                reduce_features, feature_keys=_KLUE_TC_FEATURE_KEYS)
        ),
        KlueConfig(
            name='tc.full',
            features=_KLUE_TC_FULL_FEATURES,
            data_url=_KLUE_TC_DATA_URL,
            description=_KLUE_TC_DESCRIPTION,
            citation="",  # The KLUE citation is sufficient.
            parsing_fn=parsing_json_examples_basic,
        ),
        KlueConfig(
            name='sts',
            features=_KLUE_STS_FEATURES,
            data_url=_KLUE_STS_DATA_URL,
            description=_KLUE_STS_DESCRIPTION,
            citation="",  # The KLUE citation is sufficient.
            parsing_fn=parsing_json_examples_basic,
            process_fn=chain(sts_cp_label, functools.partial(
                reduce_features, feature_keys=_KLUE_STS_FEATURE_KEYS))
        ),
        KlueConfig(
            name='sts.full',
            features=_KLUE_STS_FULL_FEATURES,
            data_url=_KLUE_STS_DATA_URL,
            description=_KLUE_STS_DESCRIPTION,
            citation="",  # The KLUE citation is sufficient.
            parsing_fn=parsing_json_examples_basic,
            process_fn=sts_cp_label
        ),
        KlueConfig(
            name='nli',
            features=_KLUE_NLI_FEATURES,
            data_url=_KLUE_NLI_DATA_URL,
            description=_KLUE_NLI_DESCRIPTION,
            citation="",  # The KLUE citation is sufficient.
            parsing_fn=parsing_json_examples_basic,
            process_fn=functools.partial(
                reduce_features, feature_keys=_KLUE_NLI_FEATURE_KEYS)
        ),
        KlueConfig(
            name='nli.full',
            features=_KLUE_NLI_FULL_FEATURES,
            data_url=_KLUE_NLI_DATA_URL,
            description=_KLUE_NLI_DESCRIPTION,
            citation="",  # The KLUE citation is sufficient.
            parsing_fn=parsing_json_examples_basic,
            process_fn=nli_ch_key
        ),
        KlueConfig(
            name='ner',
            features=_KLUE_NER_FULL_FEATURES,
            data_url=_KLUE_NER_DATA_URL,
            description=_KLUE_NER_DESCRIPTION,
            citation="",  # The KLUE citation is sufficient.
            parsing_fn=parsing_ner_examples
        ),
        KlueConfig(
            name='re',
            features=_KLUE_RE_FULL_FEATURES,
            data_url=_KLUE_RE_DATA_URL,
            description=_KLUE_RE_DESCRIPTION,
            citation="",  # The KLUE citation is sufficient.
            parsing_fn=parsing_json_examples_basic,
            process_fn=re_incr_entity_end_idx
        ),
        KlueConfig(
            name='dp',
            features=_KLUE_DP_FULL_FEATURES,
            data_url=_KLUE_DP_DATA_URL,
            description=_KLUE_DP_DESCRIPTION,
            citation="",  # The KLUE citation is sufficient.
            parsing_fn=parsing_dp_examples
        ),
        KlueConfig(
            name='mrc',
            features=_KLUE_MRC_FEATURES,
            data_url=_KLUE_MRC_DATA_URL,
            description=_KLUE_MRC_DESCRIPTION,
            citation="",  # The KLUE citation is sufficient.
            parsing_fn=parsing_mrc_examples
        ),
        KlueConfig(
            name='dst',
            features=_KLUE_DST_FEATURES,
            data_url=_KLUE_DST_DATA_URL,
            description=_KLUE_DST_DESCRIPTION,
            citation="",  # The KLUE citation is sufficient.
            parsing_fn=parsing_dst_examples
        ),
        KlueConfig(
            name='dst.gen',
            features=_KLUE_DST_GEN_FEATURES,
            data_url=_KLUE_DST_DATA_URL,
            description=_KLUE_DST_DESCRIPTION,
            citation="",  # The KLUE citation is sufficient.
            parsing_fn=parsing_dst_gen_examples
        ),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(klue): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=self.builder_config.features,
            homepage=_HOME_PAGE,
            citation=self.builder_config.citation + "\n" + _CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(klue): Downloads the data and defines the splits
        path_kv = {k: dl_manager.download_and_extract(
            v) for k, v in self.builder_config.data_url.items()}
        
        if self.builder_config.name == 'dst':
            with tf.io.gfile.GFile(path_kv['ontology']) as f:
                ontology = json.load(f)
                self.info._metadata = tfds.core.MetadataDict(**ontology)

        # TODO(klue): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            tfds.Split.TRAIN: self._generate_examples(path_kv=path_kv, split='train'),
            tfds.Split.TEST: self._generate_examples(path_kv=path_kv, split='dev'),
        }

    def _generate_examples(self, path_kv, split='train'):
        """Yields examples."""
        # TODO(klue): Yields (key, example) tuples from the dataset
        file_path = path_kv[split]
        gen_fn = self.builder_config.parsing_fn
        process_fn = self.builder_config.process_fn

        if self.builder_config.name=='dst.gen':
            for example in iter(gen_fn(file_path, path_kv['ontology'])):
                yield process_fn(example)

        else:
            for example in iter(gen_fn(file_path)):
                yield process_fn(example)


# tfds build --data_dir ../../../tensorflow_datasets --config tc
