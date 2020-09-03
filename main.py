import argparse
from train import Trainer

parser = argparse.ArgumentParser(description='End to End Memory Networks - PyTorch. Copyright:errdaychill')

parser.add_argument('--decay_ratio',
                    type=int,
                    default=2,
                    help='decay ratio (default : 2)')

parser.add_argument('--use_cuda',
                    action='store_true',
                    default=True,
                    help='Usage of Cuda (default : True)')

parser.add_argument('--batch_size',
                    type=int,
                    default=32,
                    help='number of batch size (default : 32) ')

parser.add_argument('--embed_dim',
                    type=int,
                    default=20,
                    help='size of embedding dimension (default : 20) ')


parser.add_argument('--data_dir',
                    type=str,
                    default="data/tasks_1-20_v1-2/en/",
                    help='data directory : "data/tasks_1-20_v1-2/en"' )

parser.add_argument('--task_id',
                    type=int,
                    default=1,
                    help='task number (default : 1)')

parser.add_argument('--learning_rate',
                    type=int,
                    default=0.01,
                    help='learning rate (default : 0.01)')

parser.add_argument('--num_hop',
                    type=int,
                    default=3,
                    help='number of hops (default : 3)')

parser.add_argument('--epoch',
                    type=int,
                    default=100,
                    help='training epoch (default : 100)')

config = parser.parse_args()

if __name__=='__main__':
    tr = Trainer(config)
    tr.progress()
