import torch
import torch.nn as ann
import torch.F as F
import math
import nltk
import numpy as np
from torch.autograd import Variable

def softmax(x):
    # axis = 1 
    f = x - np.max(x)
    smax = np.exp(f) / np.exp(f).sum()
    return smax




class MemN2NHop(nn.Module):
    def __init__(self, context, query, vocab_size=, mem_dim=100, internal_dim=100, embed_dim=100):
        super.__init__()
        self.context = 
        self.query = 
        self.A = Variable(torch.tensor(vocab_size, embed_dim), requires_grad=True)
        self.A = nn.Linear(vocab_size, embed_dim)
        self.B = Variable(torch.tensor(vocab_size, embed_dim), requires_grad=True)
        self.B = nn.Linear(vocab_size, embed_dim)
        self.C = Variable(torch.tensor(vocab_size, embed_dim, requires_grad=True)
        self.C = nn.Linear(vocab_size, embed_dim)

    def forward(x):
        memory_vec = self.A(self.context)
        internal_state = self.B(self.query)
        


