from utils import logger
from dataset.data_handlers import get_dataloaders, DataHandler
from dataset.datasets import  DatasetDuplet1
def dataloader(dataset,batch_size):
    data_handler_class = DataHandler
    ds_train_class = DatasetDuplet1
    ds_query_class = DatasetDuplet1
    ds_db_class = DatasetDuplet1

    images,captions,labels = get_dataloaders(data_handler_class, ds_train_class, ds_query_class, ds_db_class,dataset,batch_size)
    return images,captions,labels
