# -*- coding: utf-8 -*- 

"""aihub dataset."""

import tensorflow_datasets as tfds
import aihub

class AiHubTest(tfds.testing.DatasetBuilderTestCase):
      
  DATASET_CLASS = aihub.AIHub
  SPLITS = {
      'train': 1,  # Number of fake train example
      'validation': 1,  # Number of fake test example
  }

if __name__ == '__main__':
    tfds.testing.test_main()