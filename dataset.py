import os
import random
from itertools import chain #python 내 모듈. 데이터 분석 시 무언가 반복되는 것에대한 처리에 용이하당 
import numpy as np
import torch
from torch.utils.data import Dataset
from data_utils import loadTask, vectorizeData

# vocab & word2idx 구축
# sentence, story, query, memory 각각 크기 비교 및 정의
# 데이터 벡터화 후 데이터 반환(torch.LongTensor)
class bAbIDataSet(Dataset):
    def __init__(self, data_dir, task_id=1, memory_size=100, train=True):
        super().__init__()
        self.data_dir = data_dir
        self.task_id = task_id

        # load the data    
        train_data, test_data = loadTask(self.data_dir, self.task_id)
        data = train_data + test_data

        # 단어들 정수화 --> vocab 구축
        self.vocab = set()
        for story, query, answer in data:
            self.vocab = self.vocab | set(list(chain.from_iterable(story)) + query + answer)
        self.vocab = sorted(self.vocab)
        word_idx = {word:i+1 for i, word in enumerate(self.vocab)}
        #word_idx = dict((word, i+1) for i, word in enumerate(self.vocab))

        self.max_story_size = max([len(story) for story, _, _ in data])
        self.max_query_size = max([len(query) for _, query, _ in data])
        self.max_sentence_size = max([len(sen) for sen in chain.from_iterable([story for story, _, _ in data])])
        self.memory_size = min(memory_size, self.max_story_size)

        # add time words/indices. whY???????????? 
        for i in range(self.memory_size):
            word_idx["time{}".format(i+1)] = "time{}".format(i+1)

        self.num_vocab = len(word_idx) + 1 # <NIL> 까지 + 1 
        self.max_sentence_size = max(self.max_query_size, self.max_sentence_size) # for the position
        self.max_sentence_size += 1 # +1 for time words (TE 인덱스)
        self.word_idx = word_idx 

        self.mean_story_size = int(np.mean([len(s) for s, _, _ in data]))

        # vectorize된 data corpus 반환. 
        if train:
            story, query, answer = vectorizeData(train_data, self.word_idx, self.max_sentence_size, self.memory_size)
        else:
            story, query, answer = vectorizeData(test_data, self.word_idx, self.max_sentence_size, self.memory_size)
        self.story_data = torch.LongTensor(story)
        self.query_data = torch.LongTensor(query)
        self.answer_data = torch.LongTensor(np.argmax(answer, axis=1))

    # story_data 크기 = data size
    def __len__(self):
        return len(self.story_data)

    # story, query, answer데이터 값들을 딕셔너리 벨류값으로 반환
    def __getitem__(self, idx):
        return self.story_data[idx], self.query_data[idx], self.answer_data[idx]

