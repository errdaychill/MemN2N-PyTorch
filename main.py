import torch 
import torch.nn as nn
import argparse
import math

parser = argparse.ArgumentParser(description='End to End Memory Networks - PyTorch. Copyright:errdaychill')

parser.add_argument('--num-epoch',
                    type=int,
                    default=100,
                    help='number of iterations to train(default:100)')

parser.add_argument('--batch-size',
                    type=int,
                    default=32,
                    help='number of batch size(default:32)')

parser.add_argument('--num-hop',
                    type=int,
                    default=3,
                    help='number of hops(default:3)')

parser.add_argument('--mem-size',
                    type=int,
                    default=100,
                    help='value of memory embedding size (default:100)')

parser.add_argument('--train',
                    action='store_true',
                    default=True,
                    help='eval of train set(default:True)')

parser.add_argument('--test',
                    action='store_true'
                    default=False,
                    help='eval of test set(default:false)')

parser.add_argument('--print',
                    action='store_true',
                    default=True,
                    help='print the progress(default:True)')





