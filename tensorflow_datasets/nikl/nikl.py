"""nikl dataset."""
import os
import csv
import json
import copy
import hashlib
import functools
import unicodedata

import kss
import tensorflow as tf
import tensorflow_datasets as tfds


def _is_punctuation(char):
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


# TODO(nikl): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(nikl): BibTeX citation
_CITATION = """
"""

_VERSION = tfds.core.Version('1.0.0')


_DATASET_ROOT = {
    'cola.v1.0': 'NIKL/v1.0/CoLA',
    'dp.v1.0': 'NIKL/v1.0/DP',
    'ls.v1.0': 'NIKL/v1.0/LS',
    'messenger.v1.0': 'NIKL/v1.0/MESSENGER',
    'mp.v1.0': 'NIKL/v1.0/MP',
    'ne.v1.0': 'NIKL/v1.0/NE',
    'newspaper.v1.0': 'NIKL/v1.0/NEWSPAPER',
    'paraphrase.v1.0': 'NIKL/v1.0/PARAPHRASE',
    'spoken.v1.0': 'NIKL/v1.0/SPOKEN',
    'summarization.v1.0': 'NIKL/v1.0/SUMMARIZATION',
    'web.v1.0': 'NIKL/v1.0/WEB',
    'written.v1.0': 'NIKL/v1.0/WRITTEN',
    'ne.2020.v1.0': 'NIKL/v1.0/NE_2020',
    'cr.2020.v1.0': 'NIKL/v1.0/CR_2020',
    'za.2020.v1.0': 'NIKL/v1.0/ZA_2020',
    'dialogue.2020.v1.0': 'NIKL/v1.0/DIALOGUE_2020',
    'newspaper.2020.v1.0': 'NIKL/v1.0/NEWSPAPER_2020',
    'spoken.v1.1': 'NIKL/v1.1/SPOKEN',
}

_SPOKEN_V1_TYPO = {"principal_residence": "pricipal_residence"}

_WORD_SEQ_FEATURE = tfds.features.Sequence({
    "id": tf.int32,
    "form": tfds.features.Text(),
    "begin": tf.int32,
    "end": tf.int32,
})

_WORD_LABEL_SEQ_FEATURE = tfds.features.Sequence({
    "id": tf.int32,
    "form": tfds.features.Text(),
    "label": tfds.features.Text(),
    "word_id": tf.int32,
    "position": tf.int32,
})

# chunking methodology
# We can use IOB, IOB2, BIOES format
_NER_TAGS = [
  'PS', # PERSON
  'LC', # LOCATION
  'OG', # ORGANIZATION
  'DT', # DATE
  'TI', # TIME
  'QT', # QUANTITY
  'AF', # ARTIFACT
  'CV', # CIVILIZATION
  'AM', # ANIMAL
  'PT', # PLANT
  'FD', # STUDY_FIELD
  'TR', # THEORY
  'EV', # EVENT
  'MT', # MATERIAL
  'TM' # TERM
]

_NE_LABEL_FEATURE = tfds.features.ClassLabel(names=_NER_TAGS)

_NER_IOB2_TAGS = [
    'O',
    'B-PS', # PERSON
    'I-PS', # PERSON
    'B-LC', # LOCATION
    'I-LC', # LOCATION
    'B-OG', # ORGANIZATION
    'I-OG', # ORGANIZATION
    'B-DT', # DATE
    'I-DT', # DATE
    'B-TI', # TIME
    'I-TI', # TIME
    'B-QT', # QUANTITY
    'I-QT', # QUANTITY
    'B-AF', # ARTIFACT
    'I-AF', # ARTIFACT
    'B-CV', # CIVILIZATION
    'I-CV', # CIVILIZATION
    'B-AM', # ANIMAL
    'I-AM', # ANIMAL
    'B-PT', # PLANT
    'I-PT', # PLANT
    'B-FD', # STUDY_FIELD
    'I-FD', # STUDY_FIELD
    'B-TR', # THEORY
    'I-TR', # THEORY
    'B-EV', # EVENT
    'I-EV', # EVENT
    'B-MT', # MATERIAL
    'I-MT', # MATERIAL
    'B-TM' # TERM
    'I-TM' # TERM
]

_NE_LABEL_SEQ_FEATURE = tfds.features.Sequence({
    "id": tf.int32,
    "form": tfds.features.Text(),
    "begin": tf.int32,
    "end": tf.int32,
    "label": _NE_LABEL_FEATURE,
})

_WSD_SEQ_FEATURE = tfds.features.Sequence({
    "word": tfds.features.Text(),
    "sense_id": tf.int32,
    "pos": tfds.features.Text(),
    "begin": tf.int32,
    "end": tf.int32,
    "word_id": tf.int32,
})

_DP_LABEL_SEQ_FEATURE = tfds.features.Sequence({
    "word_id": tf.int32,
    "word_form": tfds.features.Text(),
    "head": tf.int32,
    "label": tfds.features.Text(),
    "dependent": tfds.features.Sequence(tf.int32),
})

_NE_FEATURE = tfds.features.FeaturesDict({
    "id":
        tfds.features.Text(),
    "form":
        tfds.features.Text(),
    "word":
        _WORD_SEQ_FEATURE,
    "NE":
        _NE_LABEL_SEQ_FEATURE,
})

_LS_FEATURE = tfds.features.FeaturesDict({
    "id":
        tfds.features.Text(),
    "form":
        tfds.features.Text(),
    "word":
        _WORD_SEQ_FEATURE,
    "morpheme":
        _WORD_LABEL_SEQ_FEATURE,
    "WSD":
        _WSD_SEQ_FEATURE,
})

# _MP_FEATURE = tfds.features.FeaturesDict({
#     "id":
#         tfds.features.Text(),
#     "form":
#         tfds.features.Text(),
#     "word":
#         _WORD_SEQ_FEATURE,
#     "morpheme":
#         _WORD_LABEL_SEQ_FEATURE,
# })

_MP_FEATURE = _LS_FEATURE

_DP_FEATURE = tfds.features.FeaturesDict({
    "id":
        tfds.features.Text(),
    "form":
        tfds.features.Text(),
    "DP":
        _DP_LABEL_SEQ_FEATURE,
})

_SUMMARIZATION_FEATURE = tfds.features.FeaturesDict({
    'document_id': tfds.features.Text(),
    'article': tfds.features.Text(),
    'highlights': tfds.features.Text(),
    'summary_type': tfds.features.Text(),
})

_PARAPHRASE_FEATURE = tfds.features.FeaturesDict({
    'sentence_id': tfds.features.Text(),
    'sentence_form': tfds.features.Text(),
    'paraphrases': tfds.features.Sequence({
        'form': tfds.features.Text(),
        'generation': tfds.features.Text(),
    })
})

_WRITTEN_FEATURE = tfds.features.FeaturesDict({
    'id': tfds.features.Text(),
    'title': tfds.features.Text(),
    'paragraph': tfds.features.Sequence({
        'form': tfds.features.Text(),
    })
})

_WRITTEN_PAGE_FEATURE = tfds.features.FeaturesDict({
    'id': tfds.features.Text(),
    'title': tfds.features.Text(),
    'text': tfds.features.Text(),
})

_NEWSPAPER_FEATURE = tfds.features.FeaturesDict({
    'id': tfds.features.Text(),
    'title': tfds.features.Text(),
    'topic': tfds.features.Text(),
    'original_topic': tfds.features.Text(),
    'paragraph': tfds.features.Sequence({
        'form': tfds.features.Text(),
    })
})

_NEWSPAPER_PAGE_FEATURE = tfds.features.FeaturesDict({
    'id': tfds.features.Text(),
    'title': tfds.features.Text(),
    'topic': tfds.features.Text(),
    'original_topic': tfds.features.Text(),
    'text': tfds.features.Text(),
})

_SPOKEN_SETTING_FEATURE = tfds.features.FeaturesDict({
    'relation': tfds.features.Text(),
})

_MESSENGER_SETTING_FEATURE = tfds.features.FeaturesDict({
    'relation': tfds.features.Text(),
    'intimacy': tf.int32,
    'contact_frequency': tfds.features.Text(),
})

_SPEAKER_FEATURE = tfds.features.Sequence({
    'id': tfds.features.Text(),
    'age': tfds.features.Text(),
    'occupation': tfds.features.Text(),
    'sex': tfds.features.Text(),
    'birthplace': tfds.features.Text(),
    'principal_residence': tfds.features.Text(),
    'current_residence': tfds.features.Text(),
    'education': tfds.features.Text(),
})

_SPOKEN_META_FEATURE = tfds.features.FeaturesDict({
    'topic': tfds.features.Text(),
    'speaker': _SPEAKER_FEATURE,
    'setting': _SPOKEN_SETTING_FEATURE,
})

_MESSENGER_META_FEATURE = tfds.features.FeaturesDict({
    'topic': tfds.features.Text(),
    'speaker': _SPEAKER_FEATURE,
    'setting': _MESSENGER_SETTING_FEATURE,
})

_SPOKEN_FEATURE = tfds.features.FeaturesDict({
    'id': tfds.features.Text(),
    'metadata': _SPOKEN_META_FEATURE,
    'utterance': tfds.features.Sequence({
        'form': tfds.features.Text(),
        'speaker_id': tfds.features.Text(),
    })
})

_MESSENGER_FEATURE = tfds.features.FeaturesDict({
    'id': tfds.features.Text(),
    'metadata': _MESSENGER_META_FEATURE,
    'utterance': tfds.features.Sequence({
        'form': tfds.features.Text(),
        'speaker_id': tfds.features.Text(),
    })
})

_SPOKEN_UTTERANCE_FEATURE = tfds.features.FeaturesDict({
    'id': tfds.features.Text(),
    'context': tfds.features.Sequence({
        'form': tfds.features.Text(),
        'speaker_id': tfds.features.Text(),
    }),
    'next_utterance': {
        'form': tfds.features.Text(),
        'speaker_id': tfds.features.Text(),
    },
})

_COLA_FEATURE = tfds.features.FeaturesDict({
    'idx': tf.int32,
    'sentence': tfds.features.Text(),
    'label': tfds.features.ClassLabel(
        names=["unacceptable", "acceptable"]),
})
# label_column = 'acceptability_label'

_NE2020_TAGS = [
    "PS_NAME", 
    "PS_CHARACTER", 
    "PS_PET", 
    "FD_SCIENCE", 
    "FD_SOCIAL_SCIENCE", 
    "FD_MEDICINE", 
    "FD_ART", 
    "FD_HUMANITIES", 
    "FD_OTHERS", 
    "TR_SCIENCE", 
    "TR_SOCIAL_SCIENCE", 
    "TR_MEDICINE", 
    "TR_ART", 
    "TR_HUMANITIES", 
    "TR_OTHERS", 
    "AF_BUILDING", 
    "AF_CULTURAL_ASSET", 
    "AF_ROAD", 
    "AF_TRANSPORT",
    "AF_MUSICAL_INSTRUMENT", 
    "AF_WEAPON", 
    "AFA_DOCUMENT",
    "AFA_PERFORMANCE", 
    "AFA_VIDEO", 
    "AFA_ART_CRAFT", 
    "AFA_MUSIC",
    "AFW_SERVICE_PRODUCTS", 
    "AFW_OTHER_PRODUCTS",
    "OGG_ECONOMY", 
    "OGG_EDUCATION", 
    "OGG_MILITARY", 
    "OGG_MEDIA",
    "OGG_SPORTS", 
    "OGG_ART", 
    "OGG_MEDICINE", 
    "OGG_RELIGION", 
    "OGG_SCIENCE",
    "OGG_LIBRARY", 
    "OGG_LAW", 
    "OGG_POLITICS", 
    "OGG_FOOD", 
    "OGG_HOTEL",
    "OGG_OTHERS",
    "LCP_COUNTRY", 
    "LCP_PROVINCE", 
    "LCP_COUNTY", 
    "LCP_CITY", 
    "LCP_CAPITALCITY",
    "LCG_RIVER", 
    "LCG_OCEAN", 
    "LCG_BAY", 
    "LCG_MOUNTAIN", 
    "LCG_ISLAND",
    "LCG_CONTINENT", 
    "LC_SPACE", 
    "LC_OTHERS",
    "CV_CULTURE", 
    "CV_TRIBE", 
    "CV_LANGUAGE", 
    "CV_POLICY", 
    "CV_LAW",
    "CV_CURRENCY", 
    "CV_TAX", 
    "CV_FUNDS", 
    "CV_ART", 
    "CV_SPORTS",
    "CV_SPORTS_POSITION", 
    "CV_SPORTS_INST", 
    "CV_PRIZE", 
    "CV_RELATION",
    "CV_OCCUPATION", 
    "CV_POSITION", 
    "CV_FOOD", 
    "CV_DRINK", 
    "CV_FOOD_STYLE",
    "CV_CLOTHING", 
    "CV_BUILDING_TYPE",
    "DT_DURATION", 
    "DT_DAY", 
    "DT_WEEK", 
    "DT_MONTH", 
    "DT_YEAR", 
    "DT_SEASON",
    "DT_GEOAGE", 
    "DT_DYNASTY", 
    "DT_OTHERS",
    "TI_DURATION", 
    "TI_HOUR", 
    "TI_MINUTE", 
    "TI_SECOND", 
    "TI_OTHERS",
    "QT_AGE", 
    "QT_SIZE", 
    "QT_LENGTH", 
    "QT_COUNT", 
    "QT_MAN_COUNT", 
    "QT_WEIGHT",
    "QT_PERCENTAGE", 
    "QT_SPEED", 
    "QT_TEMPERATURE", 
    "QT_VOLUME", 
    "QT_ORDER",
    "QT_PRICE", 
    "QT_PHONE", 
    "QT_SPORTS", 
    "QT_CHANNEL", 
    "QT_ALBUM", 
    "QT_ADDRESS",
    "QT_OTHERS",
    "EV_ACTIVITY", 
    "EV_WAR_REVOLUTION", 
    "EV_SPORTS", 
    "EV_FESTIVAL",
    "EV_OTHERS",
    "AM_INSECT", 
    "AM_BIRD", 
    "AM_FISH", 
    "AM_MAMMALIA", 
    "AM_AMPHIBIA",
    "AM_REPTILIA", 
    "AM_TYPE", 
    "AM_PART", 
    "AM_OTHERS",
    "PT_FRUIT", 
    "PT_FLOWER", 
    "PT_TREE", 
    "PT_GRASS", 
    "PT_TYPE", 
    "PT_PART", 
    "PT_OTHERS",
    "MT_ELEMENT", 
    "MT_METAL", 
    "MT_ROCK", 
    "MT_CHEMICAL",
    "TM_COLOR", 
    "TM_DIRECTION", 
    "TM_CLIMATE", 
    "TM_SHAPE",
    "TM_CELL_TISSUE_ORGAN", 
    "TMM_DISEASE", 
    "TMM_DRUG", 
    "TMI_HW", 
    "TMI_SW",
    "TMI_SITE", 
    "TMI_EMAIL", 
    "TMI_MODEL", 
    "TMI_SERVICE", 
    "TMI_PROJECT",
    "TMIG_GENRE", 
    "TM_SPORTS"
]
_NE2020_LABEL_FEATURE = tfds.features.ClassLabel(names=_NE2020_TAGS)


_NE2020_IOB2_TAGS = [
    "O",
    "B-PS_NAME",
    "I-PS_NAME", 
    "B-PS_CHARACTER",
    "I-PS_CHARACTER", 
    "B-PS_PET",
    "I-PS_PET", 
    "B-FD_SCIENCE",
    "I-FD_SCIENCE", 
    "B-FD_SOCIAL_SCIENCE",
    "I-FD_SOCIAL_SCIENCE", 
    "B-FD_MEDICINE",
    "I-FD_MEDICINE", 
    "B-FD_ART",
    "I-FD_ART", 
    "B-FD_HUMANITIES",
    "I-FD_HUMANITIES", 
    "B-FD_OTHERS",
    "I-FD_OTHERS", 
    "B-TR_SCIENCE",
    "I-TR_SCIENCE", 
    "B-TR_SOCIAL_SCIENCE",
    "I-TR_SOCIAL_SCIENCE", 
    "B-TR_MEDICINE",
    "I-TR_MEDICINE", 
    "B-TR_ART",
    "I-TR_ART", 
    "B-TR_HUMANITIES",
    "I-TR_HUMANITIES", 
    "B-TR_OTHERS",
    "I-TR_OTHERS", 
    "B-AF_BUILDING",
    "I-AF_BUILDING", 
    "B-AF_CULTURAL_ASSET",
    "I-AF_CULTURAL_ASSET", 
    "B-AF_ROAD",
    "I-AF_ROAD", 
    "B-AF_TRANSPORT",
    "I-AF_TRANSPORT",
    "B-AF_MUSICAL_INSTRUMENT",
    "I-AF_MUSICAL_INSTRUMENT", 
    "B-AF_WEAPON",
    "I-AF_WEAPON", 
    "B-AFA_DOCUMENT",
    "I-AFA_DOCUMENT",
    "B-AFA_PERFORMANCE",
    "I-AFA_PERFORMANCE", 
    "B-AFA_VIDEO",
    "I-AFA_VIDEO", 
    "B-AFA_ART_CRAFT",
    "I-AFA_ART_CRAFT", 
    "B-AFA_MUSIC",
    "I-AFA_MUSIC",
    "B-AFW_SERVICE_PRODUCTS",
    "I-AFW_SERVICE_PRODUCTS", 
    "B-AFW_OTHER_PRODUCTS",
    "I-AFW_OTHER_PRODUCTS",
    "B-OGG_ECONOMY",
    "I-OGG_ECONOMY", 
    "B-OGG_EDUCATION",
    "I-OGG_EDUCATION", 
    "B-OGG_MILITARY",
    "I-OGG_MILITARY", 
    "B-OGG_MEDIA",
    "I-OGG_MEDIA",
    "B-OGG_SPORTS",
    "I-OGG_SPORTS", 
    "B-OGG_ART",
    "I-OGG_ART", 
    "B-OGG_MEDICINE",
    "I-OGG_MEDICINE", 
    "B-OGG_RELIGION",
    "I-OGG_RELIGION", 
    "B-OGG_SCIENCE",
    "I-OGG_SCIENCE",
    "B-OGG_LIBRARY",
    "I-OGG_LIBRARY", 
    "B-OGG_LAW",
    "I-OGG_LAW", 
    "B-OGG_POLITICS",
    "I-OGG_POLITICS", 
    "B-OGG_FOOD",
    "I-OGG_FOOD", 
    "B-OGG_HOTEL",
    "I-OGG_HOTEL",
    "B-OGG_OTHERS",
    "I-OGG_OTHERS",
    "B-LCP_COUNTRY",
    "I-LCP_COUNTRY", 
    "B-LCP_PROVINCE",
    "I-LCP_PROVINCE", 
    "B-LCP_COUNTY",
    "I-LCP_COUNTY", 
    "B-LCP_CITY",
    "I-LCP_CITY", 
    "B-LCP_CAPITALCITY",
    "I-LCP_CAPITALCITY",
    "B-LCG_RIVER",
    "I-LCG_RIVER", 
    "B-LCG_OCEAN",
    "I-LCG_OCEAN", 
    "B-LCG_BAY",
    "I-LCG_BAY", 
    "B-LCG_MOUNTAIN",
    "I-LCG_MOUNTAIN", 
    "B-LCG_ISLAND",
    "I-LCG_ISLAND",
    "B-LCG_CONTINENT",
    "I-LCG_CONTINENT", 
    "B-LC_SPACE",
    "I-LC_SPACE", 
    "B-LC_OTHERS",
    "I-LC_OTHERS",
    "B-CV_CULTURE",
    "I-CV_CULTURE", 
    "B-CV_TRIBE",
    "I-CV_TRIBE", 
    "B-CV_LANGUAGE",
    "I-CV_LANGUAGE", 
    "B-CV_POLICY",
    "I-CV_POLICY", 
    "B-CV_LAW",
    "I-CV_LAW",
    "B-CV_CURRENCY",
    "I-CV_CURRENCY", 
    "B-CV_TAX",
    "I-CV_TAX", 
    "B-CV_FUNDS",
    "I-CV_FUNDS", 
    "B-CV_ART",
    "I-CV_ART", 
    "B-CV_SPORTS",
    "I-CV_SPORTS",
    "B-CV_SPORTS_POSITION",
    "I-CV_SPORTS_POSITION", 
    "B-CV_SPORTS_INST",
    "I-CV_SPORTS_INST", 
    "B-CV_PRIZE",
    "I-CV_PRIZE", 
    "B-CV_RELATION",
    "I-CV_RELATION",
    "B-CV_OCCUPATION",
    "I-CV_OCCUPATION", 
    "B-CV_POSITION",
    "I-CV_POSITION", 
    "B-CV_FOOD",
    "I-CV_FOOD", 
    "B-CV_DRINK",
    "I-CV_DRINK", 
    "B-CV_FOOD_STYLE",
    "I-CV_FOOD_STYLE",
    "B-CV_CLOTHING",
    "I-CV_CLOTHING", 
    "B-CV_BUILDING_TYPE",
    "I-CV_BUILDING_TYPE",
    "B-DT_DURATION",
    "I-DT_DURATION", 
    "B-DT_DAY",
    "I-DT_DAY", 
    "B-DT_WEEK",
    "I-DT_WEEK", 
    "B-DT_MONTH",
    "I-DT_MONTH", 
    "B-DT_YEAR",
    "I-DT_YEAR", 
    "B-DT_SEASON",
    "I-DT_SEASON",
    "B-DT_GEOAGE",
    "I-DT_GEOAGE", 
    "B-DT_DYNASTY",
    "I-DT_DYNASTY", 
    "B-DT_OTHERS",
    "I-DT_OTHERS",
    "B-TI_DURATION",
    "I-TI_DURATION", 
    "B-TI_HOUR",
    "I-TI_HOUR", 
    "B-TI_MINUTE",
    "I-TI_MINUTE", 
    "B-TI_SECOND",
    "I-TI_SECOND", 
    "B-TI_OTHERS",
    "I-TI_OTHERS",
    "B-QT_AGE",
    "I-QT_AGE", 
    "B-QT_SIZE",
    "I-QT_SIZE", 
    "B-QT_LENGTH",
    "I-QT_LENGTH", 
    "B-QT_COUNT",
    "I-QT_COUNT", 
    "B-QT_MAN_COUNT",
    "I-QT_MAN_COUNT", 
    "B-QT_WEIGHT",
    "I-QT_WEIGHT",
    "B-QT_PERCENTAGE",
    "I-QT_PERCENTAGE", 
    "B-QT_SPEED",
    "I-QT_SPEED", 
    "B-QT_TEMPERATURE",
    "I-QT_TEMPERATURE", 
    "B-QT_VOLUME",
    "I-QT_VOLUME", 
    "B-QT_ORDER",
    "I-QT_ORDER",
    "B-QT_PRICE",
    "I-QT_PRICE", 
    "B-QT_PHONE",
    "I-QT_PHONE", 
    "B-QT_SPORTS",
    "I-QT_SPORTS", 
    "B-QT_CHANNEL",
    "I-QT_CHANNEL", 
    "B-QT_ALBUM",
    "I-QT_ALBUM", 
    "B-QT_ADDRESS",
    "I-QT_ADDRESS",
    "B-QT_OTHERS",
    "I-QT_OTHERS",
    "B-EV_ACTIVITY",
    "I-EV_ACTIVITY", 
    "B-EV_WAR_REVOLUTION",
    "I-EV_WAR_REVOLUTION", 
    "B-EV_SPORTS",
    "I-EV_SPORTS", 
    "B-EV_FESTIVAL",
    "I-EV_FESTIVAL",
    "B-EV_OTHERS",
    "I-EV_OTHERS",
    "B-AM_INSECT",
    "I-AM_INSECT", 
    "B-AM_BIRD",
    "I-AM_BIRD", 
    "B-AM_FISH",
    "I-AM_FISH", 
    "B-AM_MAMMALIA",
    "I-AM_MAMMALIA", 
    "B-AM_AMPHIBIA",
    "I-AM_AMPHIBIA",
    "B-AM_REPTILIA",
    "I-AM_REPTILIA", 
    "B-AM_TYPE",
    "I-AM_TYPE", 
    "B-AM_PART",
    "I-AM_PART", 
    "B-AM_OTHERS",
    "I-AM_OTHERS",
    "B-PT_FRUIT",
    "I-PT_FRUIT", 
    "B-PT_FLOWER",
    "I-PT_FLOWER", 
    "B-PT_TREE",
    "I-PT_TREE", 
    "B-PT_GRASS",
    "I-PT_GRASS", 
    "B-PT_TYPE",
    "I-PT_TYPE", 
    "B-PT_PART",
    "I-PT_PART", 
    "B-PT_OTHERS",
    "I-PT_OTHERS",
    "B-MT_ELEMENT",
    "I-MT_ELEMENT", 
    "B-MT_METAL",
    "I-MT_METAL", 
    "B-MT_ROCK",
    "I-MT_ROCK", 
    "B-MT_CHEMICAL",
    "I-MT_CHEMICAL",
    "B-TM_COLOR",
    "I-TM_COLOR", 
    "B-TM_DIRECTION",
    "I-TM_DIRECTION", 
    "B-TM_CLIMATE",
    "I-TM_CLIMATE", 
    "B-TM_SHAPE",
    "I-TM_SHAPE",
    "B-TM_CELL_TISSUE_ORGAN",
    "I-TM_CELL_TISSUE_ORGAN", 
    "B-TMM_DISEASE",
    "I-TMM_DISEASE", 
    "B-TMM_DRUG",
    "I-TMM_DRUG", 
    "B-TMI_HW",
    "I-TMI_HW", 
    "B-TMI_SW",
    "I-TMI_SW",
    "B-TMI_SITE",
    "I-TMI_SITE", 
    "B-TMI_EMAIL",
    "I-TMI_EMAIL", 
    "B-TMI_MODEL",
    "I-TMI_MODEL", 
    "B-TMI_SERVICE",
    "I-TMI_SERVICE", 
    "B-TMI_PROJECT",
    "I-TMI_PROJECT",
    "B-TMIG_GENRE",
    "I-TMIG_GENRE", 
    "B-TM_SPORTS",
    "I-TM_SPORTS"
]

_NE2020_LABEL_SEQ_FEATURE = tfds.features.Sequence({
    "id": tf.int32,
    "form": tfds.features.Text(),
    "begin": tf.int32,
    "end": tf.int32,
    "label": _NE2020_LABEL_FEATURE,
})

_NE2020_FEATURE = tfds.features.FeaturesDict({
    "id":
        tfds.features.Text(),
    "form":
        tfds.features.Text(),
    "word":
        _WORD_SEQ_FEATURE,
    "ne":
        _NE2020_LABEL_SEQ_FEATURE,
})


_CR2020_FULL_FEATURE = tfds.features.FeaturesDict({
    'id': tfds.features.Text(),
    'metadata': tfds.features.FeaturesDict({
        'title': tfds.features.Text(),
        'author': tfds.features.Text(),
        'publisher': tfds.features.Text(),
        'date': tfds.features.Text(),
        'topic': tfds.features.Text(),
    }),
    'sentence': tfds.features.Sequence({
        'id': tfds.features.Text(),
        'form': tfds.features.Text(),
        'word': _WORD_SEQ_FEATURE
    }),
    'CR': tfds.features.Sequence({
        "mention": tfds.features.Sequence({
            'sentence_id': tfds.features.Text(),
            'form': tfds.features.Text(),
            'begin': tf.int32,
            'end': tf.int32,
            'NE_id': tf.int32
        })
    })
})

_CR2020_FEATURE = tfds.features.FeaturesDict({
    'id': tfds.features.Text(),
    'text': tfds.features.Text(),
    'CR': tfds.features.Sequence({
        "mention": tfds.features.Sequence(
            {
                'form': tfds.features.Text(),
                'begin': tf.int32,
                'end': tf.int32,
            }
        )
    })
})

def _base_proc(obj):
    return obj['id'], obj

def _parsing_cr(file_path):
    with tf.io.gfile.GFile(file_path, mode='r') as f:
        obj = json.loads(f.read())
        for doc in obj['document']:
            _id = doc['id']
            _metadata = doc['metadata']
            _sentence = doc['sentence']
            sent_dict = {}
            _offset = 0
            sents = []
            for v in _sentence:
                sents.append(v['form'])

                sent_dict[v["id"]] = {
                    "form": v["form"],
                    "len": len(v["form"]),
                    "offset": _offset,
                }
                _offset += len(v["form"]) + 1         

            _cr = doc['CR']

            text = ' '.join(sents)
            cr = []
            for _ments in _cr:
                ments = []
                for _ment in _ments["mention"]:
                    _sentid = _ment['sentence_id']
                    ments.append({
                        'form': _ment['form'],
                        'begin': _ment['begin'] + sent_dict[_sentid]['offset'],
                        'end': _ment['end'] + sent_dict[_sentid]['offset'],
                    })
                cr.append({
                    "mention": ments
                })
            yield _id, {
                'id': _id,
                'text': text,
                'CR': cr,
            }

# 'data' for summarization, paraphrase
# cola: tsv format


def _parsing_doc(file_path, doc_key='document'):
    with tf.io.gfile.GFile(file_path, mode='r') as f:
        obj = json.loads(f.read())
        for doc in obj[doc_key]:
            yield doc


def _parsing_para_doc(file_path, doc_key='document'):
    for obj in iter(_parsing_doc(file_path, doc_key)):
        title = obj['metadata']['title']
        for para in obj['paragraph']:
            yield para['id'], {
                'id': para['id'],
                'title': title,
                'text': para['form'].strip()
            }


def _written_para_page_proc(obj):
    title = obj['metadata']['title']
    for para in obj['paragraph']:
        yield para['id'], {
            'id': para['id'],
            'title': title,
            'text': para['form'].strip()
        }


def _page_proc(obj):
    raw_example = []
    for paragraph in obj['paragraph']:
        form = paragraph['form'].strip()
        if len(form) > 0:
            if not _is_punctuation(form[-1]):
                form += '.'
            raw_example.append(form)
    return ' '.join(raw_example)


def _paragraph_proc(obj):
    return [{'form': x['form']} for x in obj['paragraph'] if len(x['form'].strip()) > 0]


def _newspaper_proc(obj):
    return obj['id'], {
        'id': obj['id'],
        'title': obj['metadata']['title'],
        'topic': obj['metadata']['topic'],
        'original_topic': obj['metadata']['original_topic'],
        'paragraph': _paragraph_proc(obj),
    }


def _newspaper_page_proc(obj):
    return obj['id'], {
        'id': obj['id'],
        'title': obj['metadata']['title'],
        'topic': obj['metadata']['topic'],
        'original_topic': obj['metadata']['original_topic'],
        'text': _page_proc(obj),
    }


def _written_proc(obj):
    return obj['id'], {
        'id': obj['id'],
        'title': obj['metadata']['title'],
        'paragraph': _paragraph_proc(obj),
    }


def _written_page_proc(obj):
    return obj['id'], {
        'id': obj['id'],
        'title': obj['metadata']['title'],
        'text': _page_proc(obj),
    }


_SPOKEN_SETTING_TEMPLATE = {
  'relation': 'NA'
}

_MESSENGER_SETTING_TEMPLATE = {
  'relation': 'NA',
  'intimacy': 'NA',
  'contact_frequency': 'NA',
}

def _parsing_metadata(obj, dtype='spoken'):
    setting_template = _SPOKEN_SETTING_TEMPLATE if dtype=='spoken' else _MESSENGER_SETTING_TEMPLATE
    return {'topic': obj['metadata']['topic'] if 'topic' in obj['metadata'] else 'NA', 'speaker': [{
        'id': x['id'],
        'age':x['age'] if 'age' in x else 'NA',
        'occupation':x['occupation'] if 'occupation' in x else 'NA',
        'sex':x['sex'] if 'sex' in x else 'NA',
        'birthplace':x['birthplace'] if 'birthplace' in x else 'NA',
        'principal_residence':x[_SPOKEN_V1_TYPO['principal_residence']] if _SPOKEN_V1_TYPO['principal_residence'] in x else 'NA',
        'current_residence':x['current_residence'] if 'current_residence' in x else 'NA',
        'education':x['education'] if 'education' in x else 'NA',
    } for x in obj['metadata']['speaker']], 'setting': obj['metadata']['setting'] if 'setting' in obj['metadata'] else setting_template}


def _parsing_spoken(obj, dtype='spoken'):
    return obj['id'], {
        'id': obj['id'],
        'metadata': _parsing_metadata(obj, dtype=dtype),
        'utterance': [{'form': x['form'], 'speaker_id':x['speaker_id']} for x in obj['utterance'] if len(x['form'].strip()) > 0]
    }


def _reduce_utter(obj):
    reduced_utters = []
    prev_speaker = None
    utters = []
    for utterance in obj["utterance"]:
        if prev_speaker is not None:
            if prev_speaker == utterance['speaker_id']:
                if len(utterance["form"].strip()) > 0:
                    utters.append(utterance["form"].strip())
            else:
                reduced_utters.append(
                    {'form': ' '.join([x.strip() if _is_punctuation(x.strip()[-1]) else x.strip()+'.' for x in kss.split_sentences(' '.join(utters))]), 'speaker_id': prev_speaker})
                prev_speaker = utterance['speaker_id']
                utters = []
                utters.append(utterance['form'])
        elif len(utterance["form"].strip()) > 0:
            prev_speaker = utterance['speaker_id']
            utters.append(utterance['form'])
    return reduced_utters


def _create_context_utterance_pair(reduced_utters, conv_id):
    cu_pair_list = []
    for idx in range(1, len(reduced_utters)):
        cu_pair_list.append({
            'context': copy.deepcopy(reduced_utters[:idx]),
            'next_utterance': copy.deepcopy(reduced_utters[idx]),
            'id': conv_id + '.' + str(idx)
        })
    return cu_pair_list


def _parsing_spoken_utter(file_path, doc_key='document', filter_fn=lambda x: len(x['metadata']['speaker']) < 3):
    for obj in iter(_parsing_doc(file_path, doc_key)):
        if filter_fn(obj):
            conv_id = obj['id']
            #metadata = _parsing_metadata(obj)
            reduced_utters = _reduce_utter(obj)
            cu_pair_list = _create_context_utterance_pair(
                reduced_utters, conv_id)
            for cu_pair in cu_pair_list:
                yield cu_pair['id'], cu_pair


def _parsing_ls_mp_ne_dp(file_path, doc_key='document'):
    for obj in iter(_parsing_doc(file_path, doc_key)):
        for sentence in obj['sentence']:
            if len(sentence['form']) > 0:
                yield sentence['id'], sentence


def _hash_text(text):
    return hashlib.md5(tf.compat.as_text(text).encode("utf-8")).hexdigest()


def _find_fname_from_doc_dict(doc_id, doc_dict):
    doc_key = doc_id.split('.')[0]
    return doc_dict.get(doc_key, None)


def _find_id_from_doc_summarization(doc_id, fname):
    with tf.io.gfile.GFile(fname, mode='r') as f:
        doc_json = json.loads(f.read())
        for doc in doc_json['document']:
            if doc['id'] == doc_id:
                return _page_proc(doc)
    return None


def _find_id_from_sent(sent_id, fname):
    sent_id = '.'.join(sent_id.split('.')[:-1])
    doc_type = 'paragraph' if sent_id[0] == 'N' else 'utterance'
    with tf.io.gfile.GFile(fname, mode='r') as f:
        doc_json = json.loads(f.read())
        for doc in doc_json['document']:
            for sent in doc[doc_type]:
                if sent['id'] == sent_id:
                    return sent['form']
    return None


def _sentences2sentence(sentences):
    return ' '.join([x.strip() for x in sentences])


def _parsing_summary(file_path,
                     doc_dict,
                     doc_key='data',
                     summary_type=['summary_sentences', 'topic_sentences']):
    for obj in iter(_parsing_doc(file_path, doc_key)):
        doc_id = obj['document_id']
        for highlight_key in summary_type:
            hash_id = _hash_text(doc_id + highlight_key)
            fname = _find_fname_from_doc_dict(doc_id, doc_dict)
            if fname is not None:
                para = _find_id_from_doc_summarization(doc_id, fname)
                if para is not None:
                    highlights = _sentences2sentence(obj[highlight_key])
                    yield hash_id, {
                        'document_id': doc_id,
                        'article': para,
                        'highlights': highlights,
                        'summary_type': highlight_key,
                    }

def _parsing_paraphrase(file_path,
                     doc_dict,
                     doc_key='data'):
    for obj in iter(_parsing_doc(file_path, doc_key)):
        sent_id = obj['sentence_id']
        fname = _find_fname_from_doc_dict(sent_id, doc_dict)
        if fname is not None:
            sent = _find_id_from_sent(sent_id, fname)
            if sent is not None:
                yield sent_id, {
                    'sentence_id': sent_id,
                    'sentence_form': sent,
                    'paraphrases': obj['paraphrases'],
                }

def _parsing_cola(file_path):
    with tf.io.gfile.GFile(file_path, mode='r') as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for idx, row in enumerate(reader):
            #uid = row["source"] + '_' + str(idx)
            uid = idx
            yield uid, {
                "sentence": row["sentence"],
                "label": int(row["acceptability_label"]),
                "idx": uid,
            }

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

class NiklConfig(tfds.core.BuilderConfig):
    def __init__(self,
                 name,
                 data_root,
                 feature,
                 data_sp_path,
                 reading_fn,
                 parsing_fn,
                 additional_data_root=None,
                 homepage='https://corpus.korean.go.kr/',
                 split_fn=None,
                 metadata=None,
                 **kwargs):
        super(NiklConfig, self).__init__(
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




class Nikl(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for nikl dataset."""

    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    

    BUILDER_CONFIGS = [
        NiklConfig(
            name='newspaper.v1.0',
            data_root=_DATASET_ROOT['newspaper.v1.0'],
            feature=_NEWSPAPER_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_doc,
            parsing_fn=_newspaper_proc
        ),
        NiklConfig(
            name='newspaper.v1.0.page',
            data_root=_DATASET_ROOT['newspaper.v1.0'],
            feature=_NEWSPAPER_PAGE_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_doc,
            parsing_fn=_newspaper_page_proc
        ),
        NiklConfig(
            name='newspaper.v1.0.page.split',
            data_root=_DATASET_ROOT['newspaper.v1.0'],
            feature=_NEWSPAPER_PAGE_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_doc,
            parsing_fn=_newspaper_page_proc,
            split_fn=_DEFAULT_RAW_CORPUS_SPLIT,
        ),
        NiklConfig(
            name='web.v1.0',
            data_root=_DATASET_ROOT['web.v1.0'],
            feature=_WRITTEN_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_doc,
            parsing_fn=_written_proc
        ),
        NiklConfig(
            name='web.v1.0.page',
            data_root=_DATASET_ROOT['web.v1.0'],
            feature=_WRITTEN_PAGE_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_doc,
            parsing_fn=_written_page_proc
        ),
        NiklConfig(
            name='web.v1.0.page.split',
            data_root=_DATASET_ROOT['web.v1.0'],
            feature=_WRITTEN_PAGE_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_doc,
            parsing_fn=_written_page_proc,
            split_fn=_DEFAULT_RAW_CORPUS_SPLIT,
        ),
        NiklConfig(
            name='web.v1.0.paragraph_page',
            data_root=_DATASET_ROOT['web.v1.0'],
            feature=_WRITTEN_PAGE_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_para_doc,
            parsing_fn=lambda x: x,
        ),
        NiklConfig(
            name='written.v1.0',
            data_root=_DATASET_ROOT['written.v1.0'],
            feature=_WRITTEN_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_doc,
            parsing_fn=_written_proc
        ),
        NiklConfig(
            name='written.v1.0.page.split',
            data_root=_DATASET_ROOT['written.v1.0'],
            feature=_WRITTEN_PAGE_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_doc,
            parsing_fn=_written_page_proc,
            split_fn=_DEFAULT_RAW_CORPUS_SPLIT,
        ),
        NiklConfig(
            name='written.v1.0.page',
            data_root=_DATASET_ROOT['written.v1.0'],
            feature=_WRITTEN_PAGE_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_doc,
            parsing_fn=_written_page_proc
        ),
        NiklConfig(
            name='written.v1.0.paragraph_page',
            data_root=_DATASET_ROOT['written.v1.0'],
            feature=_WRITTEN_PAGE_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_para_doc,
            parsing_fn=lambda x: x,
        ),
        NiklConfig(
            name='spoken.v1.0',
            data_root=_DATASET_ROOT['spoken.v1.0'],
            feature=_SPOKEN_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['?[!E]*.json']},
            reading_fn=_parsing_doc,
            parsing_fn=functools.partial(_parsing_spoken, dtype='spoken'),
        ),
        NiklConfig(
            name='messenger.v1.0',
            data_root=_DATASET_ROOT['messenger.v1.0'],
            feature=_MESSENGER_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_doc,
            parsing_fn=functools.partial(_parsing_spoken, dtype='messenger'),
        ),
        NiklConfig(
            name='spoken.v1.0.utterance',
            data_root=_DATASET_ROOT['spoken.v1.0'],
            feature=_SPOKEN_UTTERANCE_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['?[!E]*.json']},
            reading_fn=_parsing_spoken_utter,
            parsing_fn=lambda x:x,
        ),
        NiklConfig(
            name='messenger.v1.0.utterance',
            data_root=_DATASET_ROOT['messenger.v1.0'],
            feature=_SPOKEN_UTTERANCE_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_spoken_utter,
            parsing_fn=lambda x:x,
        ),
        NiklConfig(
            name='spoken.v1.0.utterance.split',
            data_root=_DATASET_ROOT['spoken.v1.0'],
            feature=_SPOKEN_UTTERANCE_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['?[!E]*.json']},
            reading_fn=_parsing_spoken_utter,
            parsing_fn=lambda x:x,
            split_fn=_DEFAULT_DOWNSTREAMTASK_CORPUS_SPLIT,
        ),
        NiklConfig(
            name='messenger.v1.0.utterance.split',
            data_root=_DATASET_ROOT['messenger.v1.0'],
            feature=_SPOKEN_UTTERANCE_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_spoken_utter,
            parsing_fn=lambda x:x,
            split_fn=_DEFAULT_DOWNSTREAMTASK_CORPUS_SPLIT,
        ),
        NiklConfig(
            name='mp.v1.0',
            data_root=_DATASET_ROOT['mp.v1.0'],
            feature=_MP_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_ls_mp_ne_dp,
            parsing_fn=lambda x:x,
        ),
        NiklConfig(
            name='ls.v1.0',
            data_root=_DATASET_ROOT['ls.v1.0'],
            feature=_LS_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_ls_mp_ne_dp,
            parsing_fn=lambda x:x,
        ),
        NiklConfig(
            name='ne.v1.0',
            data_root=_DATASET_ROOT['ne.v1.0'],
            feature=_NE_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_ls_mp_ne_dp,
            parsing_fn=lambda x:x,
        ),
        NiklConfig(
            name='ne.2020.v1.0',
            data_root=_DATASET_ROOT['ne.2020.v1.0'],
            feature=_NE2020_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.JSON', "*.json"]},
            reading_fn=_parsing_ls_mp_ne_dp,
            parsing_fn=lambda x:x,
        ),
        NiklConfig(
            name='dp.v1.0',
            data_root=_DATASET_ROOT['dp.v1.0'],
            feature=_DP_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_ls_mp_ne_dp,
            parsing_fn=lambda x:x,
        ),
        NiklConfig(
            name='mp.v1.0.split',
            data_root=_DATASET_ROOT['mp.v1.0'],
            feature=_MP_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_ls_mp_ne_dp,
            parsing_fn=lambda x:x,
            split_fn=_DEFAULT_DOWNSTREAMTASK_CORPUS_SPLIT,
        ),
        NiklConfig(
            name='ls.v1.0.split',
            data_root=_DATASET_ROOT['ls.v1.0'],
            feature=_LS_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_ls_mp_ne_dp,
            parsing_fn=lambda x:x,
            split_fn=_DEFAULT_DOWNSTREAMTASK_CORPUS_SPLIT,
        ),
        NiklConfig(
            name='ne.v1.0.split',
            data_root=_DATASET_ROOT['ne.v1.0'],
            feature=_NE_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_ls_mp_ne_dp,
            parsing_fn=lambda x:x,
            split_fn=_DEFAULT_DOWNSTREAMTASK_CORPUS_SPLIT,
        ),
        NiklConfig(
            name='ne.2020.v1.0.split',
            data_root=_DATASET_ROOT['ne.2020.v1.0'],
            feature=_NE2020_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.JSON', "*.json"]},
            reading_fn=_parsing_ls_mp_ne_dp,
            parsing_fn=lambda x:x,
            split_fn=_DEFAULT_DOWNSTREAMTASK_CORPUS_SPLIT,
        ),
        NiklConfig(
            name='dp.v1.0.split',
            data_root=_DATASET_ROOT['dp.v1.0'],
            feature=_DP_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_ls_mp_ne_dp,
            parsing_fn=lambda x:x,
            split_fn=_DEFAULT_DOWNSTREAMTASK_CORPUS_SPLIT,
        ),
        NiklConfig(
            name='summarization.v1.0',
            data_root=_DATASET_ROOT['summarization.v1.0'],
            feature=_SUMMARIZATION_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=functools.partial(_parsing_summary, summary_type=[
                                         'summary_sentences', 'topic_sentences']),
            parsing_fn=lambda x:x,
            additional_data_root={'doc_root': [
                _DATASET_ROOT['newspaper.v1.0']+'/*.json']},
        ),
        NiklConfig(
            name='summarization.v1.0.split',
            data_root=_DATASET_ROOT['summarization.v1.0'],
            feature=_SUMMARIZATION_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=functools.partial(_parsing_summary, summary_type=[
                                         'summary_sentences', 'topic_sentences']),
            parsing_fn=lambda x:x,
            additional_data_root={'doc_root': [
                _DATASET_ROOT['newspaper.v1.0']+'/*.json']},
            split_fn=_DEFAULT_DOWNSTREAMTASK_CORPUS_SPLIT,
        ),
        NiklConfig(
            name='summarization.v1.0.summary',
            data_root=_DATASET_ROOT['summarization.v1.0'],
            feature=_SUMMARIZATION_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=functools.partial(_parsing_summary, summary_type=[
                                         'summary_sentences']),
            parsing_fn=lambda x:x,
            additional_data_root={'doc_root': [
                _DATASET_ROOT['newspaper.v1.0']+'/*.json']},
        ),
        NiklConfig(
            name='summarization.v1.0.topic',
            data_root=_DATASET_ROOT['summarization.v1.0'],
            feature=_SUMMARIZATION_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=functools.partial(_parsing_summary, summary_type=[
                                         'topic_sentences']),
            parsing_fn=lambda x:x,
            additional_data_root={'doc_root': [
                _DATASET_ROOT['newspaper.v1.0']+'/*.json']},
        ),
        NiklConfig(
            name='summarization.v1.0.summary.split',
            data_root=_DATASET_ROOT['summarization.v1.0'],
            feature=_SUMMARIZATION_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=functools.partial(_parsing_summary, summary_type=[
                                         'summary_sentences']),
            parsing_fn=lambda x:x,
            additional_data_root={'doc_root': [
                _DATASET_ROOT['newspaper.v1.0']+'/*.json']},
            split_fn=_DEFAULT_DOWNSTREAMTASK_CORPUS_SPLIT,
        ),
        NiklConfig(
            name='summarization.v1.0.topic.split',
            data_root=_DATASET_ROOT['summarization.v1.0'],
            feature=_SUMMARIZATION_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=functools.partial(_parsing_summary, summary_type=[
                                         'topic_sentences']),
            parsing_fn=lambda x:x,
            additional_data_root={'doc_root': [
                _DATASET_ROOT['newspaper.v1.0']+'/*.json']},
            split_fn=_DEFAULT_DOWNSTREAMTASK_CORPUS_SPLIT,
        ),
        NiklConfig(
            name='paraphrase.v1.0',
            data_root=_DATASET_ROOT['paraphrase.v1.0'],
            feature=_PARAPHRASE_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_paraphrase,
            parsing_fn=lambda x:x,
            additional_data_root={'doc_root': [
                _DATASET_ROOT['newspaper.v1.0']+'/*.json', _DATASET_ROOT['spoken.v1.0']+'/*.json']},
        ),
        NiklConfig(
            name='paraphrase.v1.0.split',
            data_root=_DATASET_ROOT['paraphrase.v1.0'],
            feature=_PARAPHRASE_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_paraphrase,
            parsing_fn=lambda x:x,
            additional_data_root={'doc_root': [
                _DATASET_ROOT['newspaper.v1.0']+'/*.json', _DATASET_ROOT['spoken.v1.0']+'/*.json']},
            split_fn=_DEFAULT_DOWNSTREAMTASK_CORPUS_SPLIT,
        ),
        NiklConfig(
            name='cola.v1.0',
            data_root=_DATASET_ROOT['cola.v1.0'],
            feature=_COLA_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['NIKL_CoLA_in_domain_train.tsv'],
                          tfds.Split.VALIDATION: ['NIKL_CoLA_in_domain_dev.tsv'],
                          tfds.Split.TEST: ['NIKL_CoLA_out_of_domain_dev.tsv']},
            reading_fn=_parsing_cola,
            parsing_fn=lambda x:x,
        ),
        NiklConfig(
            name='cr.2020.v1.0',
            data_root=_DATASET_ROOT['cr.2020.v1.0'],
            feature=_CR2020_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_cr,
            parsing_fn=lambda x:x,
        ),
        NiklConfig(
            name='cr.2020.full.v1.0',
            data_root=_DATASET_ROOT['cr.2020.v1.0'],
            feature=_CR2020_FULL_FEATURE,
            data_sp_path={tfds.Split.TRAIN: ['*.json']},
            reading_fn=_parsing_doc,
            parsing_fn=_base_proc,
        ),
    ]

    

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
  For the NIKL, you must manually download NIKL data from https://corpus.korean.go.kr/
  and extract it under the proper location.
  all the data have to located under manual_dir/NIKL.
  if your data not located in default manual dir, 
  you should pass the --manual_dir argument when you build this dataset.
  (e.g. tfds build --data_dir ../tensorflow_datasets --manual_dir ../path/to/dir)

  NIKL dataset consists of 13 corpora.
  This is dataset and path pairs. (all the paths are case-sensitive!)
  ============================================
  NIKL_NEWSPAPER(v1.0): manual_dir/NIKL/v1.0/NEWSPAPER/*.json
  NIKL_WRITTEN(v1.0): manual_dir/NIKL/v1.0/WRITTEN/*.json
  NIKL_WEB(v1.0): manual_dir/NIKL/v1.0/WEB/*.json

  NIKL_MESSENGER(v1.0): manual_dir/NIKL/v1.0/MESSENGER/*.json
  NIKL_SPOKEN(v1.0): manual_dir/NIKL/v1.0/SPOKEN/*.json

  NIKL_SUMMARIZATION(v1.0): manual_dir/NIKL/v1.0/SUMMARIZATION/NIKL_SC.json

  NIKL_CoLA(v1.0): manual_dir/NIKL/v1.0/CoLA/*.tsv
  NIKL_DP(v1.0): manual_dir/NIKL/v1.0/DP/NXDP190200851.json
  NIKL_MP(v1.0): manual_dir/NIKL/v1.0/MP/*.json
  NIKL_LS(v1.0): manual_dir/NIKL/v1.0/LS/*.json
  NIKL_NE(v1.0): manual_dir/NIKL/v1.0/NE/*.json
  NIKL_PARAPHRASE(v1.0): manual_dir/NIKL/v1.0/PARAPHRASE/NIKL_PC.json
  
  NIKL_NIKLex(v1.0): !!!UNSUPPORTED CORPUS!!!
  ============================================

  you can do formatting easily using the below python script.
  ============================================
  import os
  import shutil

  dirs = os.listdir('./')
  for dir_name in dirs:
    target_dir = dir_name.split('(')[0].split('_')[-1])
    shutil.move(dir_name, target_dir)
  ============================================

  If you want to know more options about tfds, type
  tfds build -h
  """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(nikl): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=self.builder_config.feature,
            homepage=self.builder_config.homepage,
            citation=_CITATION,
            metadata=self.builder_config.metadata,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path_kv = {}
        for k, v in self.builder_config.data_sp_path.items():
            path_list = []
            for vv in v:
                path_list.extend(tf.io.gfile.glob(os.path.join(
                    dl_manager.manual_dir, self.builder_config.data_root, vv)))
            path_kv[k] = path_list
            path_list = []

        for _, v in path_kv.items():
            if len(v) == 0:
                raise AssertionError("For the nikl dataset, you must manually download and extract dataset under {0}/{1}.".format(
                    dl_manager.manual_dir,
                    self.builder_config.data_root
                ))
        
        if self.builder_config.name.startswith('ne.v'):
            self.info._metadata = tfds.core.MetadataDict(ibo2=_NER_IOB2_TAGS)

        if self.builder_config.additional_data_root is not None:
            additional_data_path = []
            for v in self.builder_config.additional_data_root['doc_root']:
                additional_data_path.extend(tf.io.gfile.glob(os.path.join(dl_manager.manual_dir, v)))
            if len(additional_data_path) <= 0:
                raise AssertionError("For the summarization or paraphrase, you must manually download !!!NIKL_NEWSPAPER!!! and !!!NIKL_SPOKEN!!! corpus and extract dataset under {0}.".format(
                    dl_manager.manual_dir
                ))
            doc_dict = {os.path.splitext(os.path.basename(x))[
                0]: x for x in additional_data_path}
            self.builder_config.reading_fn = functools.partial(
                self.builder_config.reading_fn, doc_dict=doc_dict)

        if self.builder_config.split_fn is not None:
            in_files = []
            for sp_s_key in self.builder_config.split_fn['source']:
                in_files.extend(path_kv[sp_s_key])
            split_fn_kv = self.builder_config.split_fn['split']
            return {k: self._generate_examples(in_files, v) for k, v in split_fn_kv.items()}

        # TODO(nikl): Returns the Dict[split names, Iterator[Key, Example]]
        return {k: self._generate_examples(v) for k, v in path_kv.items()}

    def _generate_examples(self, path_list, split_fn=None):
        """Yields examples."""
        if split_fn is not None:
            split_filter = functools.partial(_filter_fn_hash_id, split_fn=split_fn)

        _hash_set = set()
        # TODO(nikl): Yields (key, example) tuples from the dataset
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

# tfds build --data_dir ../../tmp/tensorflow_datasets --manual_dir ../../data/raw_corpus/ko --config summarization.v1.0.summary.split
# tfds build --data_dir ../../tmp/tensorflow_datasets --manual_dir ../../data/raw_corpus/ko --config summarization.v1.0.topic.split