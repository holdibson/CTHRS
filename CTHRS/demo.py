import torch
import numpy as np
import os
import argparse
import random
from ICMR import Solver
if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
    seeds = 2023
    torch.manual_seed(seeds)
    torch.cuda.manual_seed(seeds)
    torch.cuda.manual_seed_all(seeds)
    np.random.seed(seeds)
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='rsicd', help='Dataset name: RSITMD OR RSICD OR UCM')
    parser.add_argument('--epoch', type=int, default=100, help='default:100 epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='default:256 batch_size')
    parser.add_argument('--hash_lens', type=int , default=128)
    parser.add_argument('--label_dim', type=int , default=32,help='RSICD:31,RSITMD:32,UCM:21')
    parser.add_argument('--model_dir', type=str , default='./checkpoint')
    parser.add_argument('--alpha' , default=1.5)
    parser.add_argument('--a1' ,type=float , default=0)
    parser.add_argument('--a2' , type=float , default=0.1)
    parser.add_argument('--a3' ,type=float , default=0.1)
    parser.add_argument('--device', type=str , default="cuda:0", help='cuda device')
    parser.add_argument('--task', type= int, default= 3, help="0 is train real; 1 is train hash; 2 is test real; 3 is test hash")
    config = parser.parse_args()
    Final_mean_MAP = {}
    Final_std_MAP = {}
    total_map = []
    map_i2t = []  
    map_t2i = []
    task = str(config.hash_lens)+" bits" if (config.task in [1,3]) else "real value"
    print('=============== {}--{}--Total epochs:{} ==============='.format(config.dataset, task, config.epoch))
    print('...Training is beginning...'+'---',config.hash_lens)
    solver = Solver(config)
    map, i2t, t2i = solver.train()

    