import argparse
from .base_config import BaseConfig

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', default='rsicd', help='rsitmd or rsicd', type=str)

args = parser.parse_args()

dataset = args.dataset



class ConfigModel(BaseConfig):
        # default for texts
    image_emb_for_model = "./data/image_emb_{}_.h5".format(dataset.upper())
    caption_emb_for_model = "./data/caption_emb_{}.h5".format(dataset.upper())
    dataset_json_for_model = "./data/processed_data_{}.json".format(dataset.upper())

    if dataset == 'RSITMD':
        label_dim = 32

    if dataset == 'rsicd':
        label_dim = 31

    dataset_train_split = 0.5  # part of all data, that will be used for training
    # (1 - dataset_train_split) - evaluation data
    dataset_query_split = 0.2  # part of evaluation data, that will be used for query
    # (1 - dataset_train_split) * (1 - dataset_query_split) - retrieval data

    def __init__(self, args):
        super(ConfigModel, self).__init__(args)

cfg = ConfigModel(args)
