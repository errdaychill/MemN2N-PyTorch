import os
import numpy as np
import re
# from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

#BoW implementation

def loadTask(data_dir, task_id, only_supporting=False):
    assert task_id > 0 and task_id < 21

    # 해당 디렉토리 내 파일들을 리스트 형태로 보여줌
    files = os.listdir(data_dir)

    files_path = [os.path.join(data_dir, f) for f in files] 
    s = "qa{}_".format(task_id)

    # [0]없어도 될듯. 특정 task_id에대해 유일하니까
    train_file = [f for f in files_path if s in f and 'train' in f][0]
    test_file = [f for f in files_path if s in f and 'test' in f][0]
    # 
    train_data = getStories(train_file, only_supporting)
    test_data = getStories(test_file, only_supporting)
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

# parse stories in the data file
# data엔 한 task 파일내 모든 (substory, query, answer) 튜플이 담긴다.
def parseStories(lines, only_supporting=False):
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(" ", 1) # '2 john went to the hallway' ==> ['2', 'john went to the hallway']
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

            # only_supporting=False : just store all sub stories
            else:
                substory = [x for x in story if x]
            
            data.append((substory, q, a))
            story.append("")

        # Sentences without query 
        else:
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data

# file open + close까지할려고 굳이 더 쓴듯
def getStories(file, only_supporting=False):
    with open(file) as f:
        return parseStories(f.readlines(), only_supporting=only_supporting)


def vectorizeData(data, word_idx, sentence_size, memory_size):
    S, Q, A = [], [], []
    for story, query, answer in data:
        ss = []
        for idx, sen in enumerate(story,start=1):
            pad_num = max(0, sentence_size - len(sen))
            ss.append([word_idx[w] for w in sen] + [0] * pad_num)

            # memory size만큼 가장 최근의 문장들만 slice
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


