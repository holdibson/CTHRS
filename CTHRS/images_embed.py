import os

from utils import read_json, write_hdf5, get_image_file_names, shuffle_file_names_list, reshuffle_embeddings
from configs.config_img_aug import cfg

import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import random


class ImagesDataset(torch.utils.data.Dataset):
    """
    Dataset for images
    """

    def __init__(self, image_file_names, images_folder):
        self.image_file_names = image_file_names
        self.images_folder = images_folder
        self.to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, idx):

        img = self.load_single_image(self.image_file_names[idx])
        # print('imgmode',img.mode)
        # if(img.mode=="RGBA"):
        #     print(self.image_file_names[idx])
        return idx, self.to_tensor(img)

    def __len__(self):
        return len(self.image_file_names)

    def load_single_image(self, img_name):
        """
        Load single image from the disc by name.

        :return: PIL.Image array
        """
        return Image.open(os.path.join(self.images_folder, img_name)).convert('RGB')
   
def get_embeddings(model, dataloader, device):
    """
    Get Embeddings

    :param model: model
    :param dataloader: data loader with images
    :param device: CUDA device
    :return:
    """
    with torch.no_grad():  # no need to call Tensor.backward(), saves memory
        model = model.to(device)  # to gpu (if presented)

        batch_outputs = []
        # fc_layer = torch.nn.Linear(512, 768).to(device)
        for idx, x in tqdm(dataloader, desc='Getting Embeddings (batches): '):
            x = x.to(device)
            batch_outputs.append(model(x))

        output = torch.vstack(batch_outputs)  # (batches, batch_size, output_dim) -> (batches * batch_size, output_dim)

        embeddings = output.squeeze().cpu().numpy()  # return to cpu (or do nothing), convert to numpy
        print('Embeddings shape:', embeddings.shape)
        return embeddings


def get_resnet_model_for_embedding(model=None):
    """
    Remove the last layer to get embeddings

    :param model: pretrained model (optionally)

    :return: pretrained model without last (classification layer)
    """
    if model is None:
        model = torchvision.models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    return model


if __name__ == '__main__':
    print("CREATE IMAGE EMBEDDINGS")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # device
    device = torch.device(cfg.cuda_device if torch.cuda.is_available() else "cpu")

    # read captions from JSON file
    data = read_json(cfg.dataset_json_file)

    file_names = get_image_file_names(data)
    file_names_permutated, permutations = shuffle_file_names_list(file_names)
    images_dataset = ImagesDataset(file_names_permutated, cfg.dataset_image_folder_path)
    images_dataloader = DataLoader(images_dataset, batch_size=cfg.image_emb_batch_size, shuffle=False)


    resnet = get_resnet_model_for_embedding()

    embeddings = get_embeddings(resnet, images_dataloader, device)

    embeddings_orig_order = reshuffle_embeddings(embeddings, permutations)

    write_hdf5(cfg.image_emb_file, embeddings_orig_order.astype(np.float32), 'image_emb')

    print("DONE\n\n\n")
