import itertools
import torch
import numpy as np

from utils import select_idxs


class AbstractDataset(torch.utils.data.Dataset):

    def __init__(self, images, captions, labels, idxs):
        # self.image_replication_factor = 1  # default value, how many times we need to replicate image

        self.images = images
        self.captions = captions
        self.labels = labels


        self.idxs = np.array(idxs[0])
        self.idxs_cap = np.array(idxs[1])

    def __getitem__(self, index):
        return

    def __len__(self):
        return


class DatasetDuplet1(AbstractDataset):
    """
    Class for dataset representation.

    Each image has 5 corresponding captions

    Duplet dataset sample - img-txt (image and corresponding caption)
    """

    def __init__(self, images, captions, labels, idxs):
        """
        Initialization.

        :param images: image embeddings vector
        :param captions: captions embeddings vector
        :param labels: labels vector
        """
        super().__init__(images, captions, labels, idxs)

        caption_idxs = select_idxs(len(self.captions), 1, 5)[0]
        self.captions = self.captions[caption_idxs]

    def __getitem__(self, index):
        """
        Returns a tuple (img, txt, label) - image and corresponding caption

        :param index: index of sample
        :return: tuple (img, txt, label)
        """
        return (
            torch.from_numpy(self.images[index].astype('float32')),
            torch.from_numpy(self.captions[index].astype('float32')),
            self.labels[index]
        )

    def __len__(self):
        return len(self.images)