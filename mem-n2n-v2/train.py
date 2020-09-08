import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import bAbIDataSet
from data_utils import makeVariable, vectorize
from model import MeMNN

class Trainer():
    def __init__(self, config, task_id):
        self.task_id = task_id
        self.batch_size = config.batch_size
        self.data_dir = config.data_dir
        self.memory_size = config.memory_size

        if config.use_10k:
            self.train_data = bAbIDataSet('./data/tasks_1-20_v1-2/en-10k', task_id=self.task_id, memory_size=self.memory_size)
            self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

            self.test_data = bAbIDataSet('./data/tasks_1-20_v1-2/en-10k', task_id=self.task_id, memory_size=self.memory_size, train=False)
            self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
        else:
            self.train_data = bAbIDataSet(self.data_dir, task_id=self.task_id, memory_size=self.memory_size)
            self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

            self.test_data = bAbIDataSet(self.data_dir, task_id=self.task_id, memory_size=self.memory_size, train=False)
            self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

            
        # model setting 
        setting ={
            'memory_size' : self.train_data.memory_size,
            'use_cuda' : config.use_cuda,
            'num_hop' : config.num_hop,
            'vocab_size' : self.train_data.vocab_size,
            'embed_dim' : config.embed_dim,
            'sentence_size' : self.train_data.max_sentence_size,
            'positional_encoding' : config.positional_encoding,
            'temporal_encoding' : config.temporal_encoding,
            'batch_size' : self.batch_size
        }

        # training config
        self.config = config
        self.max_epoch = config.epoch
        self.learning_rate = config.learning_rate
        self.decay_ratio = config.decay_ratio

        # model & training related configs 
        self.model = MeMNN(setting) 
        self.loss_fn = nn.CrossEntropyLoss(size_average=False)
        self.optimizer = optim.SGD(self.model.parameters(), lr=config.learning_rate)

        # cuda check
        if self.config.use_cuda :
            self.loss_fn = self.loss_fn.cuda()
            self.model = self.model.cuda()

        print("Longest sentence length", self.train_data.max_sentence_size)
        print("Longest story length", self.train_data.max_story_size)
        print("Average story length", self.train_data.mean_story_size)
        print("Number of vocab", self.train_data.vocab_size)

        self.test_acc_results = []

    def adjustLr(self, optimizer):
        for pg in optimizer.param_groups:
            pg['lr'] *= 0.5
            print('learning rate set to', pg['lr'])

    def test(self):
        correct = 0
        for s,q,a in self.test_loader:
            self.model.eval()
            
            s = makeVariable(s)
            q = makeVariable(q)
            a = makeVariable(a)

            a_pred_probs = self.model(s, q)
            a_pred = a_pred_probs.max(1)[1]

            # pred.eq().sum() => return the number of correctness
            correct += a_pred.eq(a).sum()

        acc = torch.true_divide(correct, len(self.test_loader.dataset)) * 100
        print('Task {} Test Acc : {:.2f}% -'.format(self.task_id, acc), correct, '/', len(self.test_loader.dataset))
        return acc

    def train(self):
        self.model.train()
        for epoch in range(self.max_epoch):
            correct = 0
            for s,q,a in self.train_loader:
                s = makeVariable(s)
                q = makeVariable(q)
                a = makeVariable(a)

                self.optimizer.zero_grad()

                a_pred_probs = self.model(s,q)
                a_pred = a_pred_probs.max(1)[1]
                # CrossEntropyLoss does not expect a one-hot encoded vector as the target, but class indices:
                # targets.shape : 1D
                loss = self.loss_fn(a_pred_probs, a) #(input, target)
                loss.backward()
                self.optimizer.step()

                correct += a_pred.eq(a).sum()
                acc = torch.true_divide(correct, len(self.train_loader.dataset)) * 100
            
            if epoch % 20 == 0:
                print("========================Epoch {}=======================".format(epoch))
                print('Training Acc : {:.2f}% - '.format(acc), correct, '/', len(self.train_loader.dataset))
                self.test()

            if (epoch + 1) % 25 == 0: 
                self.adjustLr(self.optimizer)
                   

    # pytorch model save & load process 
    # recommendation reason : it's possible to execute Transfer Learning
    # save as .pt/.pth

    def saveCheckPoint(self, state, is_best, filename):
        print('save model!', filename)
        torch.save(state, filename)

    def makeModelFilename(self, task_id, data_size, num_epoch):
        return '{}/Task_{}_{}-Epoch{}.model'.format('./checkpoints', task_id, data_size, num_epoch)
    
    def run(self):
        print('-----------------------------------------------')
        print('--------------------Task{}---------------------'.format(self.task_id))
        print('-----------------------------------------------')

        model_filename = self.makeModelFilename(self.task_id, self.batch_size, self.max_epoch)

        # model reload
        if os.path.isfile(model_filename) and self.config.resume:
            print("=> loading checkpoint '{}'".format(model_filename))
            checkpoint = torch.load(model_filename)
            args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at {}".format(model_filename))

        self.saveCheckPoint({
            'epoch' : self.config.epoch,
            'state_dict' : self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict()
        }, True, filename=model_filename)
      
        self.train()
        print('Final Acc')
        acc = self.test()
        self.test_acc_results.append(acc)

    def result(self):
        for i, acc in enumerate(self.test_acc_results):
            print('Task {}: Acc {:.2f}%'.format(i+1, acc))
    
                

                
                
           

