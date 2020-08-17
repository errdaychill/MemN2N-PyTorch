import os
import numpy as np
import re
# from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

#BoW implementation


def loadTask(data_dir, task_id):
    assert task_id > 0 and task_id < 21

    files = os.listdir(data_dir)
    files_path = [os.path.join(data_dir, f) for f in files] 
    s = "qa{}_".format(task_id)
    # [0]없어도 될듯. 유일하니까
    train_file = [f for f in files_path if s in f and 'train' in f][0]
    test_file = [f for f in files_path if s in f and 'test' in f][0]
    train_data = getStories(train_file)
    test_data = getStories(test_file)
    return train_data, test_data

def tokenize(sentence):
    """
    tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    """
    # str.strip() : 문장 양 끝의 \n과 공백 제거
    # re.split() : str.strip()과 다르게 더 중구난방한 문장들도 스플릿 가능
    #   +: 앞에 문자가 1번 이상 등장
    #   ?: 앞의 문자가 0번 또는 1번만 등장
    return [sent.strip() for sent in re.split("(\W+)?", sent) if sent.strip()]


def parseStories(lines, only_supporting=False):
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(" ", 1) # 'john went to the hallway' ==> ['2', 'john went to the hallway']
        nid = int(nid)
        if nid == 1:
            story = []
        
        # find the query
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
                sup_fact = map(int, sup_fact.split()) # 왜 스플릿?
                substory = [story[i - 1] for i in sup_fact]
            
            # only_supporting=False : just store all sub stories
            else:
                substory = [x for x in story if x]
            
            
            data.append((substory, q, a))
            story.append("")

        # rest sentences
        #             
        else:
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data

def getStories(file, only_supporting):
    with open(file) as f:
        return parseStories(f.readlines(), only_supporting=only_supporting)


def vectorize(data, word_idx, sentence_size, memory_size):
    S, Q, A = [], [], []
    for story, query, answer in data:
        ss = []
        for idx, sen in enumerate(story,start=1):
            pad_num = max(0, sentence_size - len(sen))
            ss.append([word_idx[w] for w in sen] + [0] * pad_num)

            # 가장 최근 memory_size개의 문장들만 memory에 저장
            ss = ss[::-1][:memory_size][::-1]

            # Make the last word of each sentence the time 'word' which
            # corresponds to vector of lookup table
            for i in range(len(ss)):
                ss[i][-1] = len(word_idx) - memory_size - i + len(ss)
            
            # pad to memory_size
            lm = max(0, memory_size - len(ss))
            for _ in range(lm):
                ss.append([0] * sentence_size)
            
            lq = max(0, sentence_size - len(query))
            q = [word_idx[w] for w in query] + [0] * lq
            
            # 0 reserved for <NIL>
            y = np.zeros(len(word_idx) + 1)
            for a in answer:
                y[word_idx[a]] = 1

            S.append(ss)
            Q.append(q)
            A.append(y)

    return np.array(S), np.array(Q), np.array(A)


