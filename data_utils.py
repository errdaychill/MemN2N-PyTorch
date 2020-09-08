import os
import numpy as np
import re
from nltk.tokenize import word_tokenize
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
    train_data = parseStories(train_file, only_supporting)
    test_data = parseStories(test_file, only_supporting)
    return train_data, test_data


def tokenize(sentence):
    """
    - return the tokens of a sentence including punctuation - 
    tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    """
    # str.strip() : 문장 양 끝의 \n과 공백 제거
    # re.split() : str.strip()과 다르게 더 중구난방한 문장들도 스플릿 가능
    #   +: 앞에 문자가 1번 이상 등장
    #   ?: 앞의 문자가 0번 또는 1번만 등장
    words = [w.strip() for w in word_tokenize(sentence) if w.strip()]
    if words[-1] is '?' or '.':
        return words[:-1]

# parse stories in the data file
# data엔 한 task 파일내 모든 (substory, query, answer) 튜플이 담긴다.
# 구두점 / 물음표 제거 필히
def parseStories(file, only_supporting=False):
    with open(file) as f:
        data = []
        story = []
        for line in f.readlines():
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
            
                # only_supporting=True : 정답과 직결되는 문장만 보관 
                if only_supporting:
                    sup_fact = map(int, sup_fact.split()) 
                    substory = [story[i - 1] for i in sup_fact] 

                # only_supporting=False : 한 query에대한 모든 서브문장들을 substory로
                # ...if x] 로 공백을 제외하고 순수한 context 문장들만 substory로.
                # data1 = (문장2개, q1, a1)
                # data2 = (문장4개, q2, a2)
                # ..
                # data5 = (문장10개, q5, a5)
                # ---story reset---
                # data6 = (문장2개, q6 a6)
                # ..
                else:
                    substory = [x for x in story if x]
           
                # data는 (substory, q, a) 튜플형태.
                # query에 대한 substory는 그 query가 나오기 전 까지 등장하는 다른 query이외 모든 문장들
                data.append((substory, q, a))
                # 공백을 넣어줌으로써 얻는 효과? 단지 구분용? 구분용이라기엔 substory에 공백도 포함되므로 실제로 구분해주는 역할을 하는 건 아닌거같다.
                story.append("")
            # Sentences without query = Story sentences
            else:
                sent = tokenize(line)
                story.append(sent)
        return data




def vectorizeData(data, word_idx, sentence_size, memory_size):
    S, Q, A = [], [], []
    for story, query, answer in data:
        # ss.size() => memory_size로 pad됨.
        ss = []
        for idx, sen in enumerate(story, 1):
            pad_num = max(0, sentence_size - len(sen))
            ss.append([word_idx[w] for w in sen] + [0] * pad_num)
            # memory size만큼 가장 최근의 문장들만 slice
        ss = ss[::-1][:memory_size][::-1]

            # Make the last word of each sentence the time 'word' which
            # corresponds to vector of lookup table
            # 템포럴 인코딩(TE)?
            # len(ss)는 왜 더하는지 몰겠다
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

        # S = (query 수, memory_size, max_sentence_size)
        # Q = (max_sentence_size, )
        # A = (|vocab|, )
        S.append(ss); Q.append(q); A.append(y)
    return np.array(S), np.array(Q), np.array(A)


