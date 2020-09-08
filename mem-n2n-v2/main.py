import argparse
from train import Trainer

parser = argparse.ArgumentParser(description='End to End Memory Networks - PyTorch. Copyright:errdaychill')

# data 
parser.add_argument('--data_dir',
                    type=str,
                    default='./data/tasks_1-20_v1-2/en',
                    help='data directory - 1k training egs')

                    
parser.add_argument('--use_10k',
                    action='store_true',
                    default=False,
                    help='use 10k training dataset (default : False)')

# model config 
parser.add_argument('--memory_size',
                    type=int,
                    default=50,
                    help='size of memory (default : 50) ')

parser.add_argument('--embed_dim',
                    type=int,
                    default=20,
                    help='size of embedding dimension (default : 20) ')


parser.add_argument('--num_hop',
                    type=int,
                    default=3,
                    help='number of hops (default : 3)')

# training
parser.add_argument('--epoch',
                    type=int,
                    default=100,
                    help='training epoch (default : 100)')

parser.add_argument('--learning_rate',
                    type=int,
                    default=0.01,
                    help='learning rate (default : 0.01)')

parser.add_argument('--decay_ratio',
                    type=int,
                    default=2,
                    help='decay ratio (default : 2)')

parser.add_argument('--batch_size',
                    type=int,
                    default=32,
                    help='number of batch size (default : 32) ')

parser.add_argument('--resume',
                    action='store_true',
                    default=False,
                    help='read pretrained models (default : False)')
# mode
parser.add_argument('--use_cuda',
                    action='store_true',
                    default=True,
                    help='Usage of Cuda (default : True)')

parser.add_argument('--positional_encoding',
                    action='store_true',
                    default=True,
                    help='use positional encoding (default : True)')

parser.add_argument('--temporal_encoding',
                    action='store_true',
                    default=True,
                    help='use temporal encoding (default : True)')

config = parser.parse_args()

if __name__=='__main__':
    for number in range(1,21):
        tr = Trainer(config, number)
        tr.run()
    tr.result()
