import json
import os
import random
import logging

import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from collections import defaultdict
from tools.config import load_config
from tools.utils import cache_or, download_zip, Timer

logger = logging.getLogger()


class KGDataset(Dataset):
    def __init__(self, config, share_in_search=False, mode=None):
        self.share_in_search = share_in_search
        self.config = config
        self.folder = os.path.join(os.path.dirname(__file__), "data", self.raw_folder)

        self.mode = config.getx("dataset/split_mode", "leave-one-out") if mode is None else mode

        self.device = config.getx("device", "cuda")
        self.batch_size = config.getx("DataLoader/batch_size", 1024)
        self.train, self.valid, self.tests, self.e_num, self.r_num = self.load_data()

        self.train_size = len(self.train)
        self.valid_size = len(self.valid)

        self.candidate_num = config.getx("candidate_num", 10)

        self._hr2t = None

        # self.valid_candidates = cache_or(f"cache.{self.raw_folder}.candidate.pkl", folder=self.folder,
        #                                  generator=lambda: self._get_candidate(self.valid),
        #                                  abort_if_not_exist="grid_spec" in self.config)
        #
        # self.tests_candidates = cache_or(f"cache.{self.raw_folder}.candidate.pkl", folder=self.folder,
        #                                  generator=lambda: self._get_candidate(self.tests),
        #                                  abort_if_not_exist="grid_spec" in self.config)
        self.valid_tensor = [x.squeeze(1) for x in torch.tensor(self.valid, device=self.device).split([1, 1, 1], dim=1)]
        self.tests_tensor = [x.squeeze(1) for x in torch.tensor(self.tests, device=self.device).split([1, 1, 1], dim=1)]
        # self.tests_candidates_tensor = torch.tensor(self.tests_candidates, device=self.device)
        # self.valid_candidates_tensor = torch.tensor(self.valid_candidates, device=self.device)
        # self.candidates = cache_or(f"cache.{self.raw_folder}.candidate.pkl", folder=self.folder,
        #                            generator=lambda: self._get_candidates(),
        #                            abort_if_not_exist="grid_spec" in self.config)
        self.candidates = self._get_candidates()
        self.candidates_tensor = self.candidates.to(device=self.device)

        config['e_num'] = self.e_num
        config['r_num'] = self.r_num
        logger.warning(f'e_num:{self.e_num}')
        logger.warning(f'r_num:{self.r_num}')

    #################################################################################
    # disable the hr2t to make the training easy
    # @property
    # def hr2t(self):
    #     if self._hr2t is None:
    #         self._hr2t = cache_or(f"cache.{self.raw_folder}.hr2t.pkl", folder=self.folder,
    #                               generator=lambda: self._get_hr2t(),
    #                               abort_if_not_exist="grid_spec" in self.config)
    #     return self._hr2t
    # def _get_candidate(self, datas):
    #     all_entity_set = set(range(self.e_num))
    #     return [
    #         random.choices(list(all_entity_set - self.hr2t[(h, r)]),
    #                        k=self.candidate_num)
    #         for h, r, t in tqdm(datas, desc='candidates')
    #     ]
    #
    #################################################################################

    def _get_candidates(self):
        return torch.randint(0, self.e_num, [self.valid_size, self.candidate_num])

    def _get_hr2t(self):
        hr2t = defaultdict(set)
        for h, r, t in tqdm(self.train, desc='hr2t'):
            hr2t[(h, r)].add(t)
        return hr2t

    def __getitem__(self, index):
        h, r, p = self.train[index]
        n = random.randint(0, self.e_num - 1)
        while n in self.hr2t[(h, r)]:
            n = random.randint(0, self.e_num - 1)
        return h, r, p, n

    def __len__(self):
        return len(self.train)

    @property
    def raw_folder(self):
        return "AB219Kdc_KG"

    @property
    def raw_file(self):
        return "AB219Kdc-kg.txt"

    @property
    def sep(self):
        return '\t'

    def fetch_dataset(self, rating_file):
        url = "https://github.com/JianhuanZhuo/dataset-package/releases/download/v0.0.1/" \
              "AB219Kdc-kg.txt.zip"
        download_zip(url, os.path.dirname(rating_file))

    def load_data(self):
        random.seed(0)
        return cache_or(f"cache.{self.raw_folder}.split.pkl", folder=self.folder,
                        generator=lambda: self.load_raw(),
                        abort_if_not_exist="grid_spec" in self.config)

    def load_raw(self):
        rating_file = os.path.join(self.folder, self.raw_file)
        if not os.path.exists(rating_file):
            if os.path.exists(rating_file + ".zip"):
                import zipfile
                with zipfile.ZipFile(rating_file + ".zip", 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(rating_file))
            else:
                self.fetch_dataset(rating_file)

        sps = []
        with open(rating_file, "r", encoding='utf-8') as fp:
            for line in fp:
                sp = line.strip().split(self.sep)
                if len(sp) != 3:
                    continue
                sps.append(sp)

        e_set = sorted(list({e for h, r, t in sps for e in (h, t)}))
        r_set = sorted(list({r for h, r, t in sps}))
        remap_e = {e: x for x, e in enumerate(e_set)}
        remap_r = {r: x for x, r in enumerate(r_set)}

        with open(os.path.join(self.folder, "remap_e.json"), 'w') as fp:
            json.dump(remap_e, fp)
        with open(os.path.join(self.folder, "remap_r.json"), 'w') as fp:
            json.dump(remap_r, fp)

        triplets_remapped = [
            (remap_e[h], remap_r[r], remap_e[t])
            for h, r, t in sps
        ]

        if self.mode == "leave-one-out":
            train, valid, tests = self.split_random_by_proportion(triplets_remapped)
        elif self.mode == "all-for-train":
            train, valid, tests = self.all_for_train(triplets_remapped)
        else:
            raise NotImplementedError(f"mode: {self.mode}")
        return train, valid, tests, len(remap_e), len(remap_r)

    @staticmethod
    def split_random_by_proportion(data):
        size = len(data)
        with Timer(f"shuffle dataset:{len(data)}"):
            random.shuffle(data)
        test_size = int(size*0.01)
        # return data[:int(size * 0.8)], data[int(size * 0.8):int(size * 0.9)], data[int(size * 0.9):]
        return data[test_size*2:], data[:test_size], data[test_size:test_size*2]

    @staticmethod
    def all_for_train(data):
        return data, [], []

    # @staticmethod
    # def split_proportion(data):
    #     """
    #     按数据集中 user 做分割
    #     """
    #     train = []
    #     valid = []
    #     tests = []
    #
    #     for u, vs in group_kv(data).items():
    #         vs = list(vs)
    #         size = len(vs)
    #         random.shuffle(vs)
    #         for v in vs[:int(size * 0.8)]:
    #             train.append((u, v))
    #         for v in vs[int(size * 0.8):int(size * 0.9)]:
    #             valid.append((u, v))
    #         for v in vs[int(size * 0.9):]:
    #             tests.append((u, v))
    #
    #     return train, valid, tests


class BSKGDataset(KGDataset):
    def __init__(self, config, share_in_search=False):
        super().__init__(config, share_in_search)
        with Timer("load train_tensor to GPU"):
            self.train_tensor = torch.tensor(self.train, device=self.device)

    def index_train(self, idx):
        return self.train_tensor[idx]

    def __getitem__(self, index):
        # h, r, p = self.train[index]
        # n = random.randint(0, self.e_num)
        # while n in self.hr2t[(h, r)]:
        #     n = random.randint(0, self.e_num)
        # return h, r, p, n
        idx = torch.randint(0, self.train_size, [self.batch_size], device='cuda')
        # train_data = self.train[idx]
        train_data = self.index_train(idx)
        h, r, p = train_data.split([1, 1, 1], dim=1)
        h = h.squeeze(dim=1)
        r = r.squeeze(dim=1)
        p = p.squeeze(dim=1)
        # n = torch.randint(0, self.e_num, [self.batch_size], device='cuda')
        n = h
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
    cfg = load_config("../config.yaml")
    dataset1 = KGDataset(config=cfg)
