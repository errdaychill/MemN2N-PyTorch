import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import bAbIDataSet
from model import MemN2N

class Trainer():
    def __init__(self, config):

        # data load
        self.train_data = bAbIDataSet(config.data_dir, config.task_id)
        self.train_loader = DataLoader(self.train_data, batch_size=config.batch_size, num_workers=1, shuffle=True)
        self.test_data = bAbIDataSet(config.data_dir, config.task_id, train=False)
        self.test_loader = DataLoader(self.test_data, batch_size=config.batch_size, num_workers=1, shuffle=False)

        # model setting 
        setting ={
            'use_cuda' : config.use_cuda,
            'num_hop' : config.num_hop,
            'vocab_size' : self.train_data.num_vocab,
            'embed_dim' : config.embed_dim,
            'sentence_size' : self.train_data.max_sentence_size
        }

        # training config
        self.max_epoch = config.epoch
        self.model = MemN2N(setting) 
        self.loss_fn = nn.CrossEntropyLoss(size_average=False)
        self.optimizer = optim.SGD(self.model.parameters(), lr=config.learning_rate)
        self.learning_rate = config.learning_rate
        self.decay_ratio = config.decay_ratio
        
        self.config = config

        # cuda check
        if self.config.use_cuda:
            self.loss_fn = self.loss_fn.cuda()
            self.model = self.model.cuda()

        print("Longest sentence length", self.train_data.max_sentence_size)
        print("Longest story length", self.train_data.max_story_size)
        print("Average story length", self.train_data.mean_story_size)
        print("Number of vocab", self.train_data.num_vocab)


    def trainSingleEpoch(self, epoch):
        for step, (story, query, answer) in enumerate(self.train_loader):
            story = Variable(story)
            query = Variable(query)
            answer = Variable(answer)
            
            if self.config.use_cuda:
                story = story.cuda()
                query = query.cuda()
                answer = answer.cuda()
            
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(story, query)[0], answer) 
            loss.backward()
            self.optimizer.step()

        return loss.data
            

    def progress(self):
        for epoch in range(self.max_epoch):
            if epoch % 25 == 0:
                self.decayLearningRate(self.optimizer, self.learning_rate)
            loss = self.trainSingleEpoch(epoch)

            if (epoch + 1) % 10 == 0:
                train_acc = self.evaluate('train')
                test_acc = self.evaluate('test')
                print('epoch: ', epoch, 'loss', loss, 'train_acc: ', train_acc, 'test_acc: ', test_acc)
        print('train_acc: ', train_acc, 'test_acc: ', test_acc)
    

    def evaluate(self, data='train'):
        correct = 0
        loader = self.test_loader if data == 'test' else self.train_loader
        for story, query, answer in loader:
            story = Variable(story)
            query = Variable(query)
            answer = Variable(answer)
        
            if self.config.use_cuda:
                story = story.cuda()
                query = query.cuda()
                answer = answer.cuda()
                
            # Variable.data == old fashioned member. access to Variable's underlying Tensor Ver=0.4.0 이후 삭제
            pred_answer_prob = self.model(story, query)[1] 
            pred = pred_answer_prob.data.max(1)[1] #max func return (max, argmax)
            #pred = pred_answer_prob.data.argmax(axis=1)
            correct += pred.eq(answer.data).cpu().sum()
        
        acc = torch.true_divide(correct, len(loader.dataset))
        return acc

    def decayLearningRate(self, optimizer, lr):
        decay_ratio = self.decay_ratio
        lr /= decay_ratio
        for param_groups in self.optimizer.param_groups:
            param_groups['lr'] = lr
          
        
