import logging
import os
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from tqdm import tqdm

from tools.config import load_config
from tools.utils import cache, exist_or_download

logger = logging.getLogger()


# BBM_hop1_connect_kg_dataset.py

# http://az-eus-p40-6-worker-10049.eastus.cloudapp.azure.com:48846/notebooks/mycontainer/dataset/coldstart/
# OnlyBridge/%E8%BF%87%E6%BB%A4%E5%87%BA%20hop1-connect%20%E6%95%B0%E6%8D%AE%E9%9B%86.ipynb
class BBMhop1ConnectKGDataset(Dataset):
    def __init__(self, config, split_mode='all_for_train'):
        self.config = config
        self.folder = os.path.join(os.path.dirname(__file__), "data", self.raw_folder)
        self.candidate_num = config.getx("dataset/candidate_num", 100)
        self.device = config.getx("device", "cuda")

        self.epochs = config.getx("epochs", 500)
        self.batch_size = config.getx("DataLoader/batch_size", 1024)

        self.split_mode = split_mode if split_mode is not None else self.config.getx("dataset/split_mode",
                                                                                     "all_for_train").replace('-', '_')

        self.cache_folder = os.path.join(os.path.dirname(__file__), "data", self.raw_folder, "cache")
        self.cache_abort_if_not_exist = "grid_spec" in self.config
        self.cache_identify_str = f"{self.raw_folder}-{self.candidate_num}-{self.split_mode}"

        self.train, self.valid, self.tests, self.e_map, self.r_map = self.load_data()
        self.num_entity = len(self.e_map)
        self.num_relation = len(self.r_map)
        self.train_size = len(self.train)
        self.train_tensor = torch.tensor(self.train, device=self.device)
        config['num_entity'] = self.num_entity
        config['num_relation'] = self.num_relation

    @property
    def kg(self):
        return self._get_kg_of_train()

    @cache
    def _get_kg_of_train(self):
        hs = [h for h, r, t in self.train]
        ts = [t for h, r, t in self.train]
        return [hs, ts]

    @property
    def raw_folder(self):
        return "BBM_hop1_connect"

    @property
    def raw_file(self):
        return "BBM_hop1_connect.csv"

    @property
    def sep(self):
        return '\t'

    @property
    def record_count(self):
        return 4615600

    @property
    def dataset_url(self):
        return "https://github.com/JianhuanZhuo/dataset-package/releases/download/v0.0.2/" \
               "BBM_hop1_connect.csv.zip"

    @cache
    def load_data(self):
        random.seed(0)
        kg_remapped, e_map, r_map = self.load_raw()

        sp_fn = '_split_mode_' + self.split_mode
        if hasattr(self, sp_fn):
            train, valid, tests = getattr(self, sp_fn)(kg_remapped)
        else:
            raise NotImplementedError(f"split_mode: {self.split_mode}")
        # train.sort()
        return train, valid, tests, e_map, r_map

    @cache
    def load_raw(self):
        kg_triplet_file = os.path.join(self.folder, self.raw_file)
        exist_or_download(kg_triplet_file, download_url=self.dataset_url)
        sps = []
        with tqdm(total=self.record_count, desc='load_raw') as pbar:
            with open(kg_triplet_file) as infile:
                while True:
                    lines = infile.readlines(1024 ** 2 * 10)
                    if not lines:
                        break
                    c = 0
                    for line in lines:
                        sps.append(line.strip().split(self.sep))
                        c += 1
                    pbar.update(c)
        if hasattr(self, 'record_count'):
            assert len(sps) == self.record_count
        remap_e = {
            e: x
            for x, e in enumerate(tqdm(sorted(list({
                e
                for h, r, t in tqdm(sps, desc='e_set')
                for e in (h, t)
            })), desc='remap_e'))}
        remap_r = {
            r: x
            for x, r in enumerate(tqdm(sorted(list({
                r
                for h, r, t in tqdm(sps, desc='r_set')
            })), desc='remap_r'))}

        remapped_kg = [
            (remap_e[h], remap_r[r], remap_e[t])
            for h, r, t in tqdm(sps, desc='remapped_kg')
        ]
        return remapped_kg, remap_e, remap_r

    @staticmethod
    def _split_mode_all_for_train(data):
        random.shuffle(data)
        return data, [], []

    def index_train(self, idx):
        return self.train_tensor[idx]

    def __getitem__(self, index):
        idx = torch.randint(0, self.train_size, [self.batch_size], device=self.device)
        # train_data = self.train[idx]
        train_data = self.index_train(idx)
        h, r, p = train_data.split([1, 1, 1], dim=1)
        h = h.squeeze(dim=1)
        r = r.squeeze(dim=1)
        p = p.squeeze(dim=1)
        n = torch.randint(0, self.num_entity, [self.batch_size], device='cuda')
        # n = h
        assert h.shape == r.shape == p.shape == n.shape == torch.Size([self.batch_size])
        return h, r, p, n

    def __len__(self):
        return len(self.train) // self.batch_size


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
    dataset1 = BBMhop1ConnectKGDataset(config=cfg, split_mode='all_for_train')
    if dataset1.kg:
        pass
