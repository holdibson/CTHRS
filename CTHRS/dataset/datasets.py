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
#         if caption_idxs and len(caption_idxs) >1200:  # 确保 caption_idxs 非空且长度 > 318
#             index = [1398, 1372, 1390, 1371 ,1348 ,1377 ,1374, 1343, 1347 ,1444, 1443, 1350, 1386, 1373,
#  1310 ,1355, 1420 ,1346, 1416, 1397]
#             selected_captions = [caption_idxs[i] for i in index]  
#             print("idx_",selected_captions)
#         else:
#             print("Index 318 is out of bounds!")
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