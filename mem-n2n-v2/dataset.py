import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from data_utils import loadTask, vectorize
from itertools import chain 
from functools import reduce

# vocab & word2idx 구축
# sentence, story, query, memory 각각 크기 비교 및 정의
# 데이터 벡터화 후 데이터 반환(torch.LongTensor)

class bAbIDataSet(Dataset):
    def __init__(self, data_dir, task_id=1, memory_size=50, train=True):
        super().__init__()
        self.data_dir = data_dir
        self.task_id = task_id

        self.train_data, self.test_data = loadTask(self.data_dir, self.task_id, only_supporting=False)
        # train_data, self.test_data, vocab_dict = loadData
        self.data = self.train_data + self.test_data 

        # size info
        self.max_story_size = max([len(s) for s,_,_ in self.data])
        self.max_query_size = max([len(q) for _,q,_ in self.data])
        self.max_sentence_size = max([len(sen) for s,_,_ in self.data for sen in s])
        self.mean_story_size = int(np.mean([len(s) for s,_,_ in self.data]))
        
        # 왜 아직도 min인지 몰랑
        self.memory_size = min(memory_size, self.max_story_size)

        self.vocab = set() 
        for s,q,a, in self.data:
            self.vocab |= set(list(chain.from_iterable(s)) + q + a)
        # self.vocab = sorted(reduce(lambda x,y : x|y, set(list(chain.from_iterable(s)) + q + a) for s, q, a in self.data))
        self.vocab = sorted(self.vocab)
        self.word2idx = {w:i for i, w in enumerate(self.vocab,1)}
        self.word2idx['<PAD>'] = 0

        self.vocab_size = len(self.vocab) + 1

        if train:
            story, query, answer = vectorize(self.train_data, self.word2idx, self.max_story_size, 
            self.max_sentence_size, self.max_query_size)
        else:
            story, query, answer = vectorize(self.test_data, self.word2idx, self.max_story_size, 
            self.max_sentence_size, self.max_query_size)

        self.story =  torch.LongTensor(story)
        self.query =  torch.LongTensor(query)
        self.answer = torch.LongTensor(answer)

    def __len__(self):
        return len(self.story)

    def __getitem__(self, index):
        return self.story[index] , self.query[index], self.answer[index]
        

