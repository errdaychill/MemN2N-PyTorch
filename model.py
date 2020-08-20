import torch
import torch.nn as nn
import torch.F as F
import math
import nltk
import numpy as np
from torch.autograd import Variable

def softmax(x, axis=0):
    f = x - np.max(x,axis=axis)
    smax = np.exp(f) / np.exp(f).sum(axis=axis)
    return smax


class MemN2NHop(nn.Module):
    def __init__(self, vocab_size, mem_size=100, sentence_size=10, batch_size=32, mem_dim=100, internal_dim=100, embed_dim=100):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_batch = batch_size
        self.mem_size = mem_size
        self.sen_size = sentence_size

        self.A = Variable(torch.tensor(mem_size, embed_dim), requires_grad=True)
        self.sen_to_mem_embed = nn.init.normal_(self.A,std=0.1)

        self.B = Variable(torch.tensor(sentence_size, embed_dim), requires_grad=True)
        self.query_embed = nn.init.normal_(self.B, std=0.1)

        self.C = Variable(torch.tensor(vocab_size, embed_dim, requires_grad=True)
        self.sen_to_out_embed = nn.init.normal_(self.C, std=0.1)

        self.W = Variable(torch.tensor(vocab_size, embed_dim, requires_grad=True)
        self.final_weight = nn.init.normal_(self.W, std=0.1)

        self.t_A = Variable(torch.tensor(vocab_size, embed_dim), requires_grad=True)
        self.t_C = Variable(torch.tensor(vocab_size, embed_dim), requires_grad=True)
    
    # context(story) : (batch_size, mem_size)
    # query : (batch_size, sen_size)
    # answer : (vocab_size, )
    def forward(self, context, query):
        memory_vec = context.mm(self.sen_to_mem_embed) # (batch_size, embed_dim)
        memory_vec = memory_vec.unsqueeze(1) #(batch_size, 1, embed_dim)

        internal_vec = query.mm(self.query_embed) # (batch_size, embed_dim)
        internal_vec = internal_vec.unsqueeze(2) #(batch_sizE, embed_dim ,1)

        output_vec = context.mm(self.sen_to_out_embed) # (batch_size, embed_dim)

        p_i = softmax(internal_vec.bmm(memory_vec))  #(batch_size, 1,1)
        p_i = p_i.squeeze(1) #(batch_size, 1)

        # response_vec : 'o' in the paper
        response_vec = np.sum(output_vec * p_i, axis=0) # (1, embed_dim)
        
        y_hat = 
        






        

