import logging
import os
import random

from torch.utils.data import Dataset
from tqdm import tqdm

from tools.config import load_config
from tools.utils import cache, exist_or_download
from dataset.BBM_hop1_connect_kg_dataset import BBMhop1ConnectKGDataset

logger = logging.getLogger()


# BBM_hop1_connect_kg_dataset.py

# http://az-eus-p40-6-worker-10049.eastus.cloudapp.azure.com:48846/notebooks/mycontainer/dataset/coldstart/
# OnlyBridge/%E8%BF%87%E6%BB%A4%E5%87%BA%20hop1-connect%20%E6%95%B0%E6%8D%AE%E9%9B%86.ipynb
class MBMhop1ConnectKGDataset(BBMhop1ConnectKGDataset):
    @property
    def raw_folder(self):
        return "MBM-hop1-connect"

    @property
    def raw_file(self):
        return "MBM-hop1-connect.csv"

    @property
    def record_count(self):
        return 7396359

    @property
    def dataset_url(self):
        return "https://github.com/JianhuanZhuo/dataset-package/releases/download/v0.0.2/" \
               "MBM-hop1-connect.csv.zip"


if __name__ == '__main__':
    """
    #user : 15897
    #item : 39768
    #train: 194417
    #valid: 15897
    #tests: 15897
    """
    logging.basicConfig(level=logging.INFO)
    cfg = load_config("../config.yaml")
    dataset1 = MBMhop1ConnectKGDataset(config=cfg, split_mode='all_for_train')
    if dataset1.kg:
        pass
