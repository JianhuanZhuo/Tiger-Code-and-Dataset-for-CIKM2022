import json
import os
import random

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.space import sampling_resolver
from dataset.space.util import u2is
from tools.config import load_config
from tools.utils import group_kv, cache, Timer


class ML100KDataset(Dataset):
    def __init__(self, config, split_mode=None, **kwargs):
        self.config = config
        self.raw_folder, self.raw_file, self.sep = self.cfg_name()
        self.folder = os.path.join(os.path.dirname(__file__), "data", self.raw_folder)
        self.cache_folder = os.path.join(os.path.dirname(__file__), "data", self.raw_folder, "cache")
        self.cache_abort_if_not_exist = "grid_spec" in self.config

        self.negative_num = self.config.getx("dataset/negative_num", 0)
        self.split_mode = split_mode if split_mode is not None else self.config.getx("dataset/split_mode",
                                                                                     "leave_one_out").replace('-', '_')
        self.sampling_with_proportion = self.config.getx("dataset/sampling_with_proportion", False)

        self.rating_filter = config.getx("dataset/filter_rating", 4)
        self.candidate_num = config.getx("dataset/candidate_num", 100)
        self.device = config.getx("device", "cuda")
        self.epochs = config.getx("epochs", 500)
        self.batch_size = config.getx("DataLoader/batch_size", 1024)

        self.cache_identify_str = f"{self.raw_folder}-{self.rating_filter}-{self.candidate_num}-{self.split_mode}"

        self.train, self.valid, self.tests, self.u_map, self.i_map = self.load_data()
        self.num_user = len(self.u_map)
        self.num_item = len(self.i_map)
        self.train_size = len(self.train)

        self.config['num_user'] = self.num_user
        self.config['num_item'] = self.num_item
        self.config['num_train'] = self.train_size

        if "grid_spec" not in self.config:
            self.statics_dataset()

        # self.candidates = self.generate_candidates()

        # self.full_uis = u2is(self.train + self.tests + self.valid, device=self.device)

        if split_mode == 'all_for_test':
            return

        space = config.getx("SamplingSpace", "BasedSamplingSpace")
        sampling_class = sampling_resolver.lookup(space)

        self.train_uis = group_kv(self.train)
        self.construct_u2is_list()

        with Timer(f"Construct {space}"):
            self.sampling = sampling_class(self.num_user, self.num_item, self.train,
                                           device=self.device,
                                           train_set=self.train_uis,
                                           cache_base=f"cache.{self.cache_identify_str}-",
                                           cache_folder=self.folder,
                                           title="enable_sampling_positive" if "grid_spec" not in self.config else None,
                                           in_grid="grid_spec" in self.config,
                                           train_full=self.train_full_indexor()
                                           )

    def construct_u2is_list(self):
        u2is_list = group_kv(self.train, return_type='list')
        assert sorted(list(u2is_list.keys()))[0] == 0
        assert sorted(list(u2is_list.keys()))[-1] == len(u2is_list) - 1
        full_u2is = []
        ux_start = []
        ux_len = []
        for u in range(len(u2is_list)):
            xs = u2is_list[u]
            ux_start.append(len(full_u2is))
            ux_len.append(len(xs))
            full_u2is += xs

        self.train_uisl = torch.tensor(full_u2is, device=self.device)
        self.ux_start = torch.tensor(ux_start, device=self.device)
        self.ux_len = torch.tensor(ux_len, device=self.device)

    def statics_dataset(self):
        print("#user :", self.num_user)
        print("#item :", self.num_item)
        print("#train:", len(self.train))
        print("#valid:", len(self.valid))
        print("#tests:", len(self.tests))

    def cfg_name(self):
        return "ML100K", "u.data", "\t"

    @cache
    def train_full_indexor(self):
        return torch.sparse_coo_tensor(
            torch.tensor(self.train).permute([1, 0]),
            torch.ones(self.train_size),
            (self.num_user, self.num_item)
        ).to_dense().bool()

    def load_data(self):
        random.seed(0)
        uis_remapped, u_map, i_map = self.load_raw()
        if hasattr(self, 'record_count'):
            assert len(uis_remapped) == self.record_count

        sp_fn = '_split_mode_' + self.split_mode
        if hasattr(self, sp_fn):
            random.seed(0)
            train, valid, tests = getattr(self, sp_fn)(uis_remapped)
        else:
            raise NotImplementedError(f"split_mode: {self.split_mode}")
        # train.sort()
        return train, valid, tests, u_map, i_map

    @cache
    def generate_candidates(self):
        assert len(self.valid) == len(self.tests)
        ground_truth = self.train + self.tests + self.valid
        u2is = group_kv(ground_truth)
        all_set = set(range(self.num_item))
        random.seed(0)
        return [
            (user, item)
            for user, _ in tqdm(self.valid, desc='candidates')
            for item in random.choices(list(all_set - u2is[user]),
                                       k=self.candidate_num)
        ]

    def __len__(self):
        return self.train_size // self.batch_size + 1

    def sample_up_random(self):
        users_positives = self.sampling.sampling_record(self.batch_size)
        users = users_positives[:, 0]
        positives = users_positives[:, 1]
        return users, positives

    def sample_up_with_proportion(self):
        users = self.sampling.sampling_users_with_proportion(self.batch_size)
        positives, pos_mask = self.sampling.sampling_positive(users, 1, return_mask=True)
        positives = positives.squeeze(1)
        pos_mask = pos_mask.squeeze(1)
        return users, (positives, pos_mask)

    def __getitem__(self, _):
        if self.sampling_with_proportion:
            users, positives = self.sample_up_with_proportion()
        else:
            users, positives = self.sample_up_random()

        negatives = self.sampling.sampling_negative(users, num=self.negative_num)
        return users, positives, negatives

    def fetch_dataset(self, rating_file):
        raise FileNotFoundError(rating_file)

    @cache
    def load_raw(self):
        rating_file = os.path.join(self.folder, self.raw_file)
        if not os.path.exists(rating_file):
            if os.path.exists(rating_file + ".zip"):
                import zipfile
                with zipfile.ZipFile(rating_file + ".zip", 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(rating_file))
            else:
                self.fetch_dataset(rating_file)
        with open(rating_file, "r") as fp:
            sps = [line.strip().split(self.sep) for line in fp]
        if hasattr(self, 'record_count'):
            assert len(sps) == self.record_count
        uis = [
            (x[0], x[1])
            for x in sps
            # if len(x) <=2 or float(x[2]) >= self.rating_filter
        ]

        # 确保有 5 条以上
        # kvs = group_kv(uis)
        # uis = [
        #     (k, v)
        #     for k, vs in kvs.items()
        #     if len(vs) >= 5
        #     for v in vs
        # ]

        u_set = sorted(list({u for u, i in uis}))
        i_set = sorted(list({i for u, i in uis}))
        remap_u = {u: x for x, u in enumerate(u_set)}
        remap_i = {i: x for x, i in enumerate(i_set)}

        with open(os.path.join(self.folder, "remap_u.json"), 'w') as fp:
            json.dump(remap_u, fp)
        with open(os.path.join(self.folder, "remap_i.json"), 'w') as fp:
            json.dump(remap_i, fp)

        uis_remapped = [
            (remap_u[u], remap_i[i])
            for u, i in uis
        ]

        return uis_remapped, remap_u, remap_i

    @staticmethod
    def _split_mode_proportion(data):
        """
        按数据集中 user 做分割
        """
        train = []
        valid = []
        tests = []

        for u, vs in group_kv(data).items():
            vs = list(vs)
            size = len(vs)
            random.shuffle(vs)
            assert len(vs) >= 3
            if len(vs) <= 5:
                sp = [1, 2]
            else:
                sp = [int(size * 0.1), int(size * 0.2)]

            for v in vs[:sp[0]]:
                tests.append((u, v))
            for v in vs[sp[0]:sp[1]]:
                valid.append((u, v))
            for v in vs[sp[1]:]:
                train.append((u, v))

        return train, valid, tests

    @staticmethod
    def _split_mode_leave_one_out(data):
        train = []
        valid = []
        tests = []

        for u, vs in group_kv(data).items():
            vs = list(vs)
            random.shuffle(vs)
            valid.append((u, vs[0]))
            tests.append((u, vs[1]))
            train += [(u, v) for v in vs[2:]]

        return train, valid, tests

    @staticmethod
    def _split_mode_cold_user(data):
        train = []
        valid = []
        tests = []

        u_group = group_kv(data)
        u_tests = set(list(u_group.keys())[:int(0.1 * len(u_group))])
        # u_valid = set(list(u_group.keys())[int(0.1 * len(u_group)):int(0.2 * len(u_group))])

        for u, vs in u_group.items():
            vs = list(vs)
            random.shuffle(vs)
            if u in u_tests:
                s = int(len(vs) * 0.5)
                tests += [(u, v) for v in vs[:s]]
                valid += [(u, v) for v in vs[s:s * 2]]
            # elif u in u_valid:
            #     valid += [(u, v) for v in vs]
            else:
                train += [(u, v) for v in vs]

        return train, valid, tests

    @staticmethod
    def _split_mode_all_for_train(data):
        random.shuffle(data)
        return data, [], []

    @staticmethod
    def _split_mode_fine_tune(data):
        train = []
        valid = []
        tests = []

        u_group = group_kv(data)
        for u, vs in u_group.items():
            vs = list(vs)
            random.shuffle(vs)

            if len(vs) > 2:
                train.append((u, vs[0]))
                vs = vs[1:]

            s = int(len(vs) * 0.5)
            tests += [(u, v) for v in vs[:s]]
            valid += [(u, v) for v in vs[s:s * 2]]

        return train, valid, tests

    @staticmethod
    def _split_mode_fine_tune_100(data):
        train = []
        valid = []
        tests = []

        u_group = group_kv(data)
        for u, vs in u_group.items():
            vs = list(vs)
            random.shuffle(vs)

            if len(vs) > 2 and len(train) < 100:
                train.append((u, vs[0]))
                vs = vs[1:]

            s = int(len(vs) * 0.5)
            tests += [(u, v) for v in vs[:s]]
            valid += [(u, v) for v in vs[s:s * 2]]

        return train, valid, tests

    @staticmethod
    def _split_mode_fine_tune_1000(data):
        train = []
        valid = []
        tests = []

        u_group = group_kv(data)
        for u, vs in u_group.items():
            vs = list(vs)
            random.shuffle(vs)

            if len(vs) > 2 and len(train) < 1000:
                train.append((u, vs[0]))
                vs = vs[1:]

            s = int(len(vs) * 0.5)
            tests += [(u, v) for v in vs[:s]]
            valid += [(u, v) for v in vs[s:s * 2]]

        return train, valid, tests

    @staticmethod
    def _split_mode_all_for_test(data):
        valid = []
        tests = []
        u_group = group_kv(data)
        for u, vs in u_group.items():
            vs = list(vs)
            random.shuffle(vs)
            s = int(len(vs) * 0.5)
            tests += [(u, v) for v in vs[:s]]
            valid += [(u, v) for v in vs[s:s * 2]]

        return [], valid, tests

    # def valid_tensor(self):
    #     return torch.tensor([u for u, i in self.valid], device=self.device), \
    #            torch.tensor([i for u, i in self.valid], device=self.device)
    #
    # def tests_tensor(self):
    #     return torch.tensor([u for u, i in self.tests], device=self.device), \
    #            torch.tensor([i for u, i in self.tests], device=self.device)
    #
    # def candi_tensor(self):
    #     return torch.tensor([u for u, i in self.candidates], device=self.device), \
    #            torch.tensor([i for u, i in self.candidates], device=self.device)
    #
    # def pack_tensor(self):
    #     vu, vi = self.valid_tensor()
    #     tu, ti = self.tests_tensor()
    #     cu, ci = self.candi_tensor()
    #
    #     if isinstance(vu, tuple):
    #         vu, vh = vu
    #         tu, th = tu
    #         cu, ch = cu
    #         return (
    #                    torch.cat([vu, tu, cu], dim=0),
    #                    torch.cat([vh, th, ch], dim=0),
    #                ), torch.cat([vi, ti, ci], dim=0)
    #     else:
    #         return torch.cat([vu, tu, cu], dim=0), torch.cat([vi, ti, ci], dim=0)

    # def pack_full_tensor(self):
    #     vu, vi = self.valid_tensor()
    #     tu, ti = self.tests_tensor()
    #
    #     if isinstance(vu, tuple):
    #         vu, vh = vu
    #         tu, th = tu
    #         return (torch.cat([vu, tu], dim=0), torch.cat([vh, th], dim=0)), torch.cat([vi, ti], dim=0)
    #     else:
    #         return torch.cat([vu, tu], dim=0), torch.cat([vi, ti], dim=0)

    def sample_items_on_users(self, users, k):
        batch_size = users.shape[0]
        res = torch.rand([batch_size, k], device=self.device)
        res *= self.ux_len[users].reshape([-1, 1])
        res = res.long()
        res += self.ux_start[users].reshape([-1, 1])
        res = self.train_uisl[res]
        return res

    def all_historical_items_on_users(self, users):
        assert users.max() < self.ux_start.shape[0]
        assert users.max() < self.ux_len.shape[0]
        assert users.min() >= 0
        users_start = self.ux_start[users]
        users_len = self.ux_len[users]
        offset = torch.cat([torch.tensor([0], device=self.device), users_len.cumsum(dim=0)[:-1]])
        assert offset.shape == users_len.shape

        us = users_start.tolist()
        ul = users_len.tolist()

        results = torch.cat([
            self.train_uisl[s:(s+l)]
            for s, l in zip(us, ul)
        ])
        return results, offset

    def full_items(self):
        return torch.arange(self.num_item, device=self.device)


if __name__ == '__main__':
    """
    #user : 942
    #item : 1447
    #train: 43929
    #valid: 5497
    #tests: 5949
    """
    cfg = load_config("../config.yaml")
    dataset = ML100KDataset(config=cfg)
