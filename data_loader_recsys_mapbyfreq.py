import os
from os import listdir
from os.path import isfile, join
import numpy as np
from tensorflow.contrib import learn
from collections import Counter
import collections

#ID is indexed by frequency, higher frequency with smaller index
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



        #if you want to use NCE loss, you have to make sure your labels are indexed by frequency as follows:
        itemfreq = {}
        self.itemrank={}
        for line in self.item:
            for item in line:
                itemfreq[item] = itemfreq.setdefault(item, 0) + 1
        sorted_x = sorted(itemfreq.items(), key=lambda kv: kv[1], reverse=True)    # <type 'list'>: [(1, 3), (2, 2), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)]
        for index, x in enumerate(sorted_x):
            self.itemrank[x[0]]=index #self.item_dict has additional 'UNK'
        #print itemrank #{1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16}
        for i,line in enumerate(self.item):
            for j, item in enumerate(line):
                newID=self.itemrank.get(item)
                self.item[i][j]=newID
        # print  self.item #start from zero













