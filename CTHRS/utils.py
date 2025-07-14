import csv
import logging
import os
import pickle
import numpy as np
import torch
from os import path as osp
import scipy.io as sio
import json
import h5py
import random
from tqdm import tqdm


def read_json(file_name, suppress_console_info=False):
    """
    Read JSON

    :param file_name: input JSON path
    :param suppress_console_info: toggle console printing
    :return: dictionary from JSON
    """
    with open(file_name, 'r') as f:
        data = json.load(f)
        if not suppress_console_info:
            print("Read from:", file_name)
    return data


def get_image_file_names(data, suppress_console_info=False):
    """
    Get list of image file names

    :param data: original data from JSON
    :param suppress_console_info: toggle console printing
    :return: list of strings (file names)
    """

    file_names = []
    for img in data['images']:
        file_names.append(img["filename"])
    if not suppress_console_info:
        print("Total number of files:", len(file_names))
    return file_names


def get_labels(data, suppress_console_info=False):
    """
    Get list of labels

    :param data: original data from JSON
    :param suppress_console_info: toggle console printing
    :return: list ints (labels)
    """

    labels = []
    for img in data['images']:
        labels.append(img["classcode"])
    if not suppress_console_info:
        print("Total number of labels:", len(labels))
    return labels


def get_captions(data, suppress_console_info=False):
    """
    Get list of formatted captions

    :param data: original data from JSON
    :return: list of strings (captions)
    """

    def format_caption(string):
        return string.replace('.', '').replace(',', '').replace('!', '').replace('?', '').lower()

    captions = []
    for img in data['images']:
        for sent in img['sentences']:
            captions.append(format_caption(sent['raw']))
    if not suppress_console_info:
        print("Total number of captions:", len(captions))
    return captions


def add_prefix_to_filename(file_name, prefix):
    """
    Adds prefix to the file name

    :param file_name: file name string
    :param prefix: prefix
    :return: file name string with prefix
    """
    bn = os.path.basename(file_name)
    dn = os.path.dirname(file_name)
    return os.path.join(dn, prefix + bn)


def write_json(file_name, data):
    """
    Write dictionary to JSON file

    :param file_name: output path
    :param data: dictionary
    :return: None
    """
    bn = os.path.basename(file_name)
    dn = os.path.dirname(file_name)
    name, ext = os.path.splitext(bn)
    file_name = os.path.join(dn, name + '.json')
    with open(file_name, 'w') as f:
        f.write(json.dumps(data, indent='\t'))
    print("Written to:", file_name)


def print_parsed_args(parsed):
    """
    Print parsed arguments

    :param parsed: Namespace of parsed arguments
    :return: None
    """
    vs = vars(parsed)
    print("Parsed arguments: ")
    for k, v in vs.items():
        print("\t" + str(k) + ": " + str(v))


def write_hdf5(out_file, data, dataset_name):
    """
    Write to h5 file

    :param out_file: file name
    :param data: data to write
    :return:
    """
    bn = os.path.basename(out_file)
    dn = os.path.dirname(out_file)
    name, ext = os.path.splitext(bn)
    out_file = os.path.join(dn, name + '.h5')
    with h5py.File(out_file, 'w') as hf:
        print("Saved as '.h5' file to", out_file)
        hf.create_dataset(dataset_name, data=data)


def read_hdf5(file_name, dataset_name, normalize=False):
    """
    Read from h5 file

    :param file_name: file name
    :param dataset_name: dataset name
    :param normalize: normalize loaded values
    :return:
    """
    with h5py.File(file_name, 'r') as hf:
        print("Read from:", file_name)
        data = hf[dataset_name][:]
        if normalize:
            data = (data - data.mean()) / data.std()
        return data


def write_pickle(path, data):
    """
    Write pickle

    :param path: path
    :param data: data to write
    :return:
    """
    dn = os.path.dirname(path)
    if not os.path.exists(dn):
        os.makedirs(dn)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def select_idxs(seq_length, n_to_select, n_from_select):
    """
    Select n_to_select indexes from each consequent n_from_select indexes from range with length seq_length, split
    selected indexes to separate arrays

    Example:

    seq_length = 20
    n_from_select = 5
    n_to_select = 2

    input, range of length seq_length:
    range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    sequences of length n_from_select:
    sequences = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]

    selected n_to_select elements from each sequence
    selected = [[0, 4], [7, 9], [13, 14], [16, 18]]

    output, n_to_select lists of length seq_length / n_from_select:
    output = [[0, 7, 13, 16], [4, 9, 14, 18]]

    :param seq_length: length of sequence, say 10
    :param n_to_select: number of elements to select
    :param n_from_select: number of consequent elements
    :return:
    """
    idxs = [[] for _ in range(n_to_select)]
    for i in range(seq_length // n_from_select):
        ints = random.sample(range(n_from_select), n_to_select)
        for j in range(n_to_select):
            idxs[j].append(i * n_from_select + ints[j])
    return idxs


def shuffle_file_names_list(file_names):
    """
    We need to shuffle the data because resnet has batch normalization

    :param file_names: img file names list
    :return: shuffled list and permutation
    """
    fn_np = np.array(file_names)
    perm = np.random.permutation(len(fn_np))
    fn_np_perm = fn_np[perm]
    return fn_np_perm, perm


def reshuffle_embeddings(embeddings, permutations):
    """
    Reorder embeddings array back to original

    :param embeddings: embeddings
    :param permutations: permutation of original shuffle
    :return: embeddings with original data order
    """

    def get_idx(i):
        idx = np.where(permutations == i)[0][0]
        return idx

    result = np.zeros_like(embeddings)

    for i in range(len(result)):
        idx = get_idx(i)
        result[i] = embeddings[idx]

    return result

def save_checkpoints(self):
    if self.task ==0:
        file_name = self.dataset + '_fusion.pth'
        ckp_path = osp.join(self.model_dir, 'real', file_name)
        obj = {
        'FusionTransformer': self.FuseTrans.state_dict()
        }
    if self.task ==1:
        file_name = self.dataset + '_hash_' + str(self.nbits)+".pth"
        ckp_path = osp.join(self.model_dir, 'hash', file_name)
        obj = {
        'FusionTransformer': self.FuseTrans.state_dict(),
        'ImageMlp': self.ImageMlp.state_dict(),
        'TextMlp': self.TextMlp.state_dict()
        }
    torch.save(obj, ckp_path)
    print('**********Save the {0} model successfully.**********'.format("real" if self.task==0 else "hash"))


def load_checkpoints(self, file_name):
    ckp_path = file_name
    try:
        obj = torch.load(ckp_path, map_location= self.device)
        print('**************** Load checkpoint %s ****************' % ckp_path)
    except IOError:
        print('********** Fail to load checkpoint %s!*********' % ckp_path)
        raise IOError
    if self.task==2:
        self.FuseTrans.load_state_dict(obj['FusionTransformer'])
    elif self.task==3:
        self.FuseTrans.load_state_dict(obj['FusionTransformer'])
        self.ImageMlp.load_state_dict(obj['ImageMlp'])
        self.TextMlp.load_state_dict(obj['TextMlp'])
def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    leng = B2.shape[1] # max inner product value
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH
def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH
def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):
    """
    :param qu_B: {-1,+1}^{mxq} query bits
    :param re_B: {-1,+1}^{nxq} retrieval bits
    :param qu_L: {0,1}^{mxl} query label
    :param re_L: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = qu_L.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, int(tsum))
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap        


def pr_curve(qB, rB, query_label, retrieval_label, tqdm_label=''):
    """
    Calculate PR curve, each point - hamming radius

    :param qB: query hash code
    :param rB: retrieval hash codes
    :param query_label: query label
    :param retrieval_label: retrieval label
    :param tqdm_label: label for tqdm's output
    :return:
    """
    if tqdm_label != '':
        tqdm_label = 'PR-curve ' + tqdm_label

    num_query = qB.shape[0]  # length of query (each sample from query compared to retrieval samples)
    num_bit = qB.shape[1]  # length of hash code
    P = torch.zeros(num_query, num_bit + 1)  # precisions (for each sample)
    R = torch.zeros(num_query, num_bit + 1)  # recalls (for each sample)
    query_label = torch.Tensor(query_label)
    retrieval_label = torch.Tensor(retrieval_label)
    qB = torch.Tensor(qB)
    rB = torch.Tensor(rB)
    # for each sample from query calculate precision and recall
    for i in tqdm(range(num_query), desc=tqdm_label):
        # gnd[j] == 1 if same class, otherwise 0, ground truth
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        # tsum (TP + FN): total number of samples of the same class
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)  # hamming distances from qB[i, :] (current sample) to retrieval samples
        # tmp[k,j] == 1 if hamming distance to retrieval sample j is less or equal to k (distance), 0 otherwise
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(hamm.device)).float()
        # total (TP + FP): total[k] is count of distances less or equal to k (from query sample to retrieval samples)
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.0001  # replace zeros with 0.1 to avoid division by zero
        # select only same class samples from tmp (ground truth masking, only rows where gnd == 1 proceed further)
        t = gnd * tmp
        # count (TP): number of true (correctly selected) samples of the same class for any given distance k
        count = t.sum(dim=-1)
        p = count / total  # TP / (TP + FP)
        r = count / tsum  # TP / (TP + FN)
        P[i] = p
        R[i] = r
    # mask to calculate P mean value (among all query samples)
    # mask = (P > 0).float().sum(dim=0)
    # mask = mask + (mask == 0).float() * 0.001
    # P = P.sum(dim=0) / mask
    # mask to calculate R mean value (among all query samples)
    # mask = (R > 0).float().sum(dim=0)
    # mask = mask + (mask == 0).float() * 0.001
    # R = R.sum(dim=0) / mask
    P = P.mean(dim=0)
    R = R.mean(dim=0)
    return P.cpu().numpy().tolist(), R.cpu().numpy().tolist()

def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH

def pr_Curve(qB, rB, query_label, retrieval_label, K):
    num_query = query_label.shape[0]
    p = [0] * len(K)
    r = [0] * len(K)
    query_label = torch.Tensor(query_label)
    retrieval_label = torch.Tensor(retrieval_label)
    qB = torch.Tensor(qB)
    rB = torch.Tensor(rB)
    for iter in range(num_query):
        gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB).squeeze()
        hamm = torch.Tensor(hamm)
        for i in range(len(K)):
            total = min(K[i], retrieval_label.shape[0])
            ind = torch.sort(hamm).indices[:int(total)]
            gnd_ = gnd[ind]
            p[i] += gnd_.sum() / total
            r[i] += gnd_.sum() / tsum
    p = torch.Tensor(p) / num_query
    r = torch.Tensor(r) / num_query
    return p, r 

def p_topK(qB, rB, query_label, retrieval_label, K ):
    num_query = query_label.shape[0]
    p = [0] * len(K)
    query_label = torch.Tensor(query_label)
    retrieval_label = torch.Tensor(retrieval_label)
    qB = torch.Tensor(qB)
    rB = torch.Tensor(rB)
    for iter in range(num_query):
        gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB).squeeze()
        hamm = torch.Tensor(hamm)
        for i in range(len(K)):
            total = min(K[i], retrieval_label.shape[0])
            ind = torch.sort(hamm).indices[:int(total)]
            gnd_ = gnd[ind]
            p[i] += gnd_.sum() / total
    p = torch.Tensor(p) / num_query
    return p
def logger():
    """
    Instantiate logger

    :return:
    """
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    log_name = 'log.txt'
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    txt_log = logging.FileHandler(os.path.join(log_dir, log_name))
    txt_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    txt_log.setFormatter(formatter)
    logger.addHandler(txt_log)

    stream_log = logging.StreamHandler()
    stream_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_log.setFormatter(formatter)
    logger.addHandler(stream_log)

    return logger          

