import argparse
from .base_config import BaseConfig

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', default='rsicd', help='rsitmd or rsicd', type=str, metavar='DATASET_NAME')

args = parser.parse_args()

dataset = args.dataset


class ConfigTxtPrep(BaseConfig):
    if dataset == 'rsitmd':
        caption_token_length = 64  # use hardcoded number, max token length for clean data is 26

    if dataset == 'rsicd':
        caption_token_length = 64  # use hardcoded number, max token length for clean data is 40

    caption_hidden_states = 4  # Number of last BERT's hidden states to use
    caption_hidden_states_operator = 'sum'  # "How to combine hidden states: 'sum' or 'concat'"

    caption_aug_rb_glove_sim_threshold = 0.65
    caption_aug_rb_bert_score_threshold = 0.75

    # caption_aug_type - augmentation method 'prob' or 'chain'
    # caption_aug_method:
    #   'prob' - sentence translated to a random lang with prob proportional to weight from captions_aug_bt_lang_weights
    #   'chain' - translates in chain en -> lang1 -> lang2 -> ... -> en (weight values are ignored)


    # available models and laguages: https://huggingface.co/Helsinki-NLP
    # en -> choice(languages) -> en; choice_probability_for_language = lang_weight / sum(lang_weights)
    caption_aug_bt_lang_weights = {'es': 1, 'de': 1}  # {lang1: lang_weight1, lang2: lang_weight2, ...}
    # caption_aug_bt_lang_weights = {'ru': 1, 'bg': 1}

    caption_aug_dataset_json = "./data/processed_data_{}.json".format(dataset.upper())
    caption_emb_file = "./data/caption_emb_{}_test.h5".format(dataset.upper())
    def __init__(self, args):
        super(ConfigTxtPrep, self).__init__(args)


cfg = ConfigTxtPrep(args)
