import argparse
from .base_config import BaseConfig

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', default='rsicd', help='ucm、rsitmd or rsicd', type=str, metavar='DATASET_NAME')

args = parser.parse_args()

dataset = args.dataset



class ConfigImgPrep(BaseConfig):
    if dataset == 'ucm':
        image_emb_batch_size = 525  
        image_folder_preload = False  
        image_aug_number = 1  

    if dataset == 'rsitmd':
        image_emb_batch_size = 200  
        image_folder_preload = False  
        image_aug_number = 1  

    if dataset == 'rsicd':
        image_emb_batch_size = 500
        image_folder_preload = False  
        image_aug_number = 1  
        
    image_emb_file = "./data/image_emb_{}_test.h5".format(dataset.upper())

    def __init__(self, args):
        super(ConfigImgPrep, self).__init__(args)


cfg = ConfigImgPrep(args)
