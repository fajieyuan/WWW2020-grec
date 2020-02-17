import os
from os import listdir
from os.path import isfile, join
import numpy as np
from tensorflow.contrib import learn
from collections import Counter
import collections

class Data_Loader:
    def __init__(self, options):

        positive_data_file = options['dir_name']
        positive_examples = list(open(positive_data_file, "r").readlines())


        # positive_examples = [s for s in positive_examples]
        self.max_document_length = max([len(x.split(",")) for x in positive_examples])
        # max_document_length = max([len(x.split()) for x in positive_examples])  #split by space, one or many, not sensitive
        vocab_processor = learn.preprocessing.VocabularyProcessor(self.max_document_length)
        self.item = np.array(list(vocab_processor.fit_transform(positive_examples)))
        self.item_dict = vocab_processor.vocabulary_._mapping


