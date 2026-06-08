import random
from torch.utils.data import DataLoader
from utils import read_hdf5, read_json, get_labels,select_idxs
import numpy as np

class DataHandler:

    def __init__(self):
        super().__init__()

    def load_train_query_db_data(self,dataset):
        """
        Load and split (train, query, db)

        :return: tuples of (images, captions, labels), each element is array
        """
        images, captions, labels = load_dataset(dataset)

        train, query, db = self.split_data(images, captions, labels)

        return train, query, db

    @staticmethod
    def split_data(images, captions, labels):
        """
        Split dataset to get training, query and db subsets

        :param: images: image embeddings array
        :param: captions: caption embeddings array
        :param: labels: labels array

        :return: tuples of (images, captions, labels), each element is array
        """
        idx_tr, idx_q, idx_db = get_split_idxs(len(images))
        idx_tr_cap, idx_q_cap, idx_db_cap = get_caption_idxs(idx_tr, idx_q, idx_db)
        
        train = images[idx_tr], captions[idx_tr_cap], labels[idx_tr], (idx_tr, idx_tr_cap)
        query = images[idx_q], captions[idx_q_cap], labels[idx_q], (idx_q, idx_q_cap)
        db = images[idx_db], captions[idx_db_cap], labels[idx_db], (idx_db, idx_db_cap)

        return train, query, db

def load_dataset(dataset):
    """
    Load dataset

    :return: images and captions embeddings, labels
    """
    images = read_hdf5("../../autodl-fs/image_emb_{}.h5".format(dataset.upper()), 'image_emb', normalize=True)
    captions = read_hdf5("../../autodl-fs/caption_emb_{}.h5".format(dataset.upper()), 'caption_emb', normalize=True)
    labels = np.array(get_labels(read_json("../../autodl-fs/processed_data_{}.json".format(dataset.upper())), suppress_console_info=True))
    
    return images, captions, labels


def get_split_idxs(arr_len):
    """
    Get indexes for training, query and db subsets

    :param: arr_len: array length

    :return: indexes for training, query and db subsets
    """
    idx_all = list(range(arr_len))
    idx_train, idx_eval = split_indexes(idx_all, 0.5)
    idx_query, idx_db = split_indexes(idx_eval, 0.2)
    return idx_train, idx_query, idx_db


def split_indexes(idx_all, split):
    """
    Splits list in two parts.

    :param idx_all: array to split
    :param split: portion to split
    :return: splitted lists
    """
    idx_length = len(idx_all)
    selection_length = int(idx_length * split)
    idx_selection = sorted(random.sample(idx_all, selection_length))

    idx_rest = sorted(list(set(idx_all).difference(set(idx_selection))))

    return idx_selection, idx_rest


def get_caption_idxs(idx_train, idx_query, idx_db):
    """
    Get caption indexes.

    :param: idx_train: train image (and label) indexes
    :param: idx_query: query image (and label) indexes
    :param: idx_db: db image (and label) indexes

    :return: caption indexes for corresponding index sets
    """
    idx_train_cap = get_caption_idxs_from_img_idxs(idx_train)
    idx_query_cap = get_caption_idxs_from_img_idxs(idx_query)
    idx_db_cap = get_caption_idxs_from_img_idxs(idx_db)
    return idx_train_cap, idx_query_cap, idx_db_cap


def get_caption_idxs_from_img_idxs(img_idxs):
    """
    Get caption indexes. There are 5 captions for each image (and label).
    Say, img indexes - [0, 10, 100]
    Then, caption indexes - [0, 1, 2, 3, 4, 50, 51, 52, 53, 54, 100, 501, 502, 503, 504]

    :param: img_idxs: image (and label) indexes

    :return: caption indexes
    """
    caption_idxs = []
    for idx in img_idxs:
        for i in range(5):  
            caption_idxs.append(idx * 5 + i)
    return caption_idxs


def get_dataloaders(data_handler, ds_train, ds_query, ds_db,dataset,batch_size):
    """
    Initializes dataloaders

    :param data_handler: data handler instance
    :param ds_train: class of train dataset
    :param ds_query: class of query dataset
    :param ds_db: class of database dataset

    :return: dataloaders
    """
    data_handler = data_handler()

    train_tuple, query_tuple, db_tuple = data_handler.load_train_query_db_data(dataset)
    dataset_triplets = ds_train(*train_tuple)
    dataloader_train = DataLoader(dataset_triplets, batch_size=batch_size, shuffle=True)
    dataset_q = ds_query(*query_tuple)
    dataloader_q = DataLoader(dataset_q, batch_size=batch_size)
    dataset_db = ds_db(*db_tuple)
    dataloader_db = DataLoader(dataset_db, batch_size=batch_size)

    return dataloader_train, dataloader_q, dataloader_db
