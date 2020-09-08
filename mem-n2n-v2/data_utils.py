import os
import numpy as np
import torch
from nltk.tokenize import word_tokenize
from itertools import chain
from functools import reduce
from torch.autograd import Variable

def makeVariable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def loadTask(data_dir, task_id, only_supporting=False):
    assert task_id > 0 and task_id < 21

    files = os.listdir(data_dir)
    files_path = [os.path.join(data_dir, file) for file in files]
    s = "qa{}_".format(task_id)

    train_file = [f for f in files_path if s in f and 'train' in f][0]
    test_file = [f for f in files_path if s in f and 'test' in f][0]
    train_data = parseStories(train_file, only_supporting)
    test_data = parseStories(test_file, only_supporting)
    return train_data, test_data


def tokenize(sentence):
    return [w.strip() for w in word_tokenize(sentence) if w.strip()]

def parseStories(file, only_supporting=False):
    with open(file) as f:
        data = []
        story = []
        for line in f.readlines():
            line = str.lower(line)
            nid, line = line.split(" ", 1) # '2 john went to the hallway' ==> ['2', 'john went to the hallway'] (두 번째 인자 = 분리할 문자 갯수)
            nid = int(nid)
            if nid == 1:
                story = []
            
            # Query
            if '\t' in line:
                q, a, sup_fact = line.split('\t')
                q = tokenize(q)
                a = [a]
                substory = None
                
                # remove question mark
                if q[-1] == '?':
                    q = q[:-1]

                # only_supporting=True : 정답과 직결되는 문장만 보관 
                if only_supporting:
                    sup_fact = map(int, sup_fact.split()) # 왜 스플릿? 숫자 하나뿐인데 split 후 map을 할 필요가 있나. 안해도 될 것 같다.
                    substory = [story[i - 1] for i in sup_fact] 
                else:
                    substory = [x for x in story if x]
                
                data.append((substory, q, a))
                story.append("")
            # Sentences without query = Story sentences
            else:
                sent = tokenize(line)
                if sent[-1] == ".":
                    sent = sent[:-1]
                story.append(sent)
    return data


def wordToIdx(sent, word2idx):
    index_vec = []
    for w in sent:
        if w in word2idx:
            index_vec.append(word2idx[w])
        else:
            index_vec.append(word2idx['<PAD>'])

    return index_vec


def vectorize(data, word2idx, story_len, max_sent_size, q_sent_size):
    vocab_size = len(word2idx)
    S, Q, A = [], [], []
    for d in data:
        tmp_story = d[0]
        story = []
        for s in tmp_story:
            sent = wordToIdx(s, word2idx)
            sent += [0] * (max_sent_size - len(sent))
            story.append(sent) 

        while len(story) < story_len:
            story.append([0] * max_sent_size)
        story = story[:story_len]

        q = wordToIdx(d[1], word2idx)
        q += [0] * (q_sent_size - len(q))
        a = wordToIdx(d[2], word2idx)
        one_hot_vec = [0] * vocab_size
        one_hot_vec[a[0]] = 1

        S.append(story)
        Q.append(q)
        A.append(a[0])
        
    return S,Q,A


