import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MeMNN(nn.Module):
    def __init__(self, setting, dropout=0.2):
        super().__init__()
        self.memory_size = setting['memory_size']
        self.embed_dim = setting['embed_dim']
        self.num_hop = setting['num_hop']
        self.use_cuda = setting['use_cuda']
        self.vocab_size = setting['vocab_size']

        self.pe_mode = setting['positional_encoding']
        self.te_mode = setting['temporal_encoding']

        self.dropout = nn.Dropout(p=dropout)
        self.A = nn.ModuleList([nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0) for _ in range(self.num_hop + 1)])

        for i in range(self.num_hop + 1):
            self.A[i].weight.data.normal_(0, 0.1)
        self.B = self.A[0]

        if self.te_mode:
            self.T_A = nn.Parameter(torch.Tensor(1,self.memory_size, self.embed_dim).normal_(0, 0.1))
            self.T_C = nn.Parameter(torch.Tensor(1,self.memory_size, self.embed_dim).normal_(0, 0.1))


    def positionalEncoding(self, sentence_size, embed_dim):
        J = sentence_size
        d = embed_dim
        if torch.cuda.is_available():
            pe_mat = Variable(torch.zeros((sentence_size, embed_dim)).cuda()) 
        for i in range(1, J+1):
            for k in range(1, d+1):
                pe_mat[i-1][k-1] = (1 - i / J) - (k/d) * (1 - 2 * i / J) #assuming 1 - based indexing
        return pe_mat

    # context:(bs, story_size, max_sent_size)
    # query:(bs, query_size)
    # d로 변환이 아니라, 차원으로 하나 더 추가됨.
    def forward(self, context, query):
        bs, story_size, max_sent_size = context.size()

        context = context.view(bs * story_size, -1) # bs * story_size, max_sent_size
        u = self.dropout(self.B(query)) # bs, query_size, embed_size 
        u = u.sum(axis=1) # bs, embed_size

        # Adjacent weight tying
        for num in range(self.num_hop):
            m_i = self.dropout(self.A[num](context)) # bs * story_size, max_sent_size, d
            m_i = m_i.view(bs, story_size, max_sent_size, -1) # bs, story_size, max_sent_size, d
            
            if self.pe_mode:
                pe_mat = self.positionalEncoding(max_sent_size, self.embed_dim)
                pe_mat = pe_mat.unsqueeze(0).unsqueeze(0) # 1,1,J,d
                pe_mat = pe_mat.repeat(bs, story_size, 1, 1) # bs, story-size, max_sent_size, d
                m_i *= pe_mat

            m_i = m_i.sum(axis=2) #bs, story_size, embed_size

            if self.te_mode and self.memory_size > story_size:
                m_i += self.T_A.repeat(bs,1,1)[:, :story_size, :]
            elif self.te_mode and self.memory_size < story_size:
                m_i[:,story_size - self.memory_size : story_size,:] += self.T_A.repeat(bs,1,1)

            c_i = self.dropout(self.A[num + 1](context)) # bs * story_size, max_sent_size, embed_size
            c_i = c_i.view(bs, story_size, max_sent_size, -1)
            c_i = c_i.sum(axis=2) #bs, story_size, embed_size

            p_i = torch.bmm(m_i, u.unsqueeze(2)).squeeze() #bs, story_size
            p_i = F.softmax(p_i, dim=1).unsqueeze(1) #bs, 1, story_size
            o = torch.bmm(p_i, c_i).squeeze(1) # bs, embed_size
            u = u + o
        
        W = torch.t(self.A[-1].weight) # (embed_size, vocab_size)
        out = torch.bmm(u.unsqueeze(1), W.unsqueeze(0).repeat(bs,1,1)).squeeze(1) #bs, vocab_size
        return F.softmax(out, dim=-1)




    
























        



                            
        

        



        

