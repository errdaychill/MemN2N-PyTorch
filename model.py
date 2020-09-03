import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import nltk
import numpy as np
from torch.autograd import Variable


def positionEncoding(sentence_size, embed_dim):
    encoding = np.ones((embed_dim, sentence_size), dtype=np.float32)
    for i in range(1, embed_dim + 1):
        for j in range(1, sentence_size+1):
            # 왜이렇게 식이..
            encoding[i-1][j-1] = (i - (embed_dim + 1) / 2) * (j - (sentence_size + 1) / 2)
            # encoding[i-1][j-1] = (1 - j / sentence_size) - ( / embed_dim)(1 - 2 * j / sentence_size)
    encoding = 1 + 4 * encoding / embed_dim / sentence_size
    # 마지막 word (time word)들은 pe 안해줌.
    encoding[:, -1] = 1.0
    return encoding.T

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        # class AttrProxy의 객체의 인자들을 반환
        # getattr(object, 'x') == object.x
        return getattr(self.module, self.prefix + str(i))

class MemN2N(nn.Module):
    def __init__(self, setting):
        super().__init__()
        
        use_cuda = setting['use_cuda']
        self.num_hop = setting['num_hop']
        vocab_size = setting['vocab_size']
        embed_dim = setting['embed_dim']
        sentence_size = setting['sentence_size']
        self.softmax = nn.Softmax()

        for h in range(self.num_hop + 1):
            C = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            C.weight.data.normal_(0, 0.1)
            # add_module(name, module): 현재 모듈에 새로운 모듈 추가
            self.add_module("C_{}".format(h), C)
        self.C = AttrProxy(self, "C_")

        # requires_grad=True : tensor에서 이뤄진 모든 연산들 말단까지 추적. backward()로 기울기 자동 계산 가능한 것.
        self.encoding = Variable(torch.FloatTensor(positionEncoding(sentence_size, embed_dim)), requires_grad=False)

        if use_cuda:
            self.encoding = self.encoding.cuda()

    def forward(self, context, query):
        context_size = context.size()
        
        u = []
        query_embed = self.C[0](query)

        encoding = self.encoding.unsqueeze(0).expand_as(query_embed)
        u.append(torch.sum(query_embed * encoding, 1))
        
        for hop in range(self.num_hop):
            embed_A = self.C[hop](context.view(context.size(0), -1))
            embed_A = embed_A.view(context_size + (embed_A.size(-1), ))
            
            encoding = self.encoding.unsqueeze(0).unsqueeze(1).expand_as(embed_A)
            m_A = torch.sum(embed_A * encoding, 2)

            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob = self.softmax(torch.sum(m_A * u_temp, 2))

            embed_C = self.C[hop + 1](context.view(context.size(0), -1))
            embed_C = embed_C.view(context_size + (embed_C.size(-1), ))
            m_C = torch.sum(embed_C * encoding, 2)

            prob = prob.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C * prob, 1)
            
            u_k = u[-1] + o_k
            u.append(u_k)

        answer_pred = u[-1]@self.C[self.num_hop].weight.transpose(0,1)
        return answer_pred, self.softmax(answer_pred)
        






        

