import argparse
from .base_config import BaseConfig

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', default='rsicd', help='rsitmd or rsicd', type=str, metavar='DATASET_NAME')
# parser.add_argument('--img-aug', default='no_aug', type=str, metavar='IMG_AUG_SET',
#                     help="image transform set: see 'image_aug_transform_sets' variable for available sets")

args = parser.parse_args()

dataset = args.dataset



class ConfigImgPrep(BaseConfig):
    if dataset == 'rsitmd':
        image_emb_batch_size = 200  # 1/4 of dataset
        image_folder_preload = False  # Load all images before running (will be loading on request otherwise)
        image_aug_number = 1  # number generated augmented images

    if dataset == 'rsicd':
        image_emb_batch_size = 500
        image_folder_preload = False  # Load all images before running (will be loading on request otherwise)
        image_aug_number = 1  # number generated augmented images
        
    image_emb_file = "./data/image_emb_{}_test.h5".format(dataset.upper())

    def __init__(self, args):
        super(ConfigImgPrep, self).__init__(args)


cfg = ConfigImgPrep(args)
