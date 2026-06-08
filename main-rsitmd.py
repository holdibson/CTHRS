import torch
import numpy as np
import os
import argparse
import random
from train import Train
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='rsitmd', help='Dataset: RSITMD OR RSICD OR UCM')
    parser.add_argument('--epoch', type=int, default=100, help='default:100 epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='default:256 batch_size')
    parser.add_argument('--bits', type=int , default=128)
    parser.add_argument('--label_dim', type=int , default=32,help='RSICD:31,RSITMD:32,UCM:21')
    parser.add_argument('--model_dir', type=str , default='./checkpoints')
    parser.add_argument('--eta' , type=float, default=1.5)
    parser.add_argument('--alpha' ,type=float , default=1.0)
    parser.add_argument('--beta' , type=float , default=0.1)
    parser.add_argument('--gama' ,type=float , default=0.1)
    parser.add_argument('--a' ,type=float , default=0.001)
    parser.add_argument('--device', type=str , default="cuda:0", help='cuda device')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task', type= int, default= 0, help="0 is train; 1 is test")
    config = parser.parse_args()
    seed = config.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    task = "train" if config.task == 0 else "test"
    print('=============== {}--{}--{}--epochs:{} ==============='.format(config.dataset, task,config.bits, config.epoch))
    train = Train(config)
    avg, i2t, t2i = train.train()
    print('final test----avg:{}, i2t:{}, t2i:{}'.format(avg,i2t,t2i))

    