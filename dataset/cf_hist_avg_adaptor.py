import random

import torch

from dataset import ML100KDataset


class CFHistAdaptor(ML100KDataset):
    def __init__(self, config, hist_dataset=None, k=None, **kwargs):
        super().__init__(config, **kwargs)
        if hist_dataset is None:
            hist_dataset = self
        self.hist_dataset = hist_dataset
        # assert self.num_user == hist_dataset.num_user

        # 需要将 target-domain 的 user 转成 source-domain 的 user 才能根据 user 获取 history
        t_s = []
        for t_user, t_uid in self.u_map.items():
            s_uid = hist_dataset.u_map[t_user]
            t_s.append([t_uid, s_uid])
        self.t2s = torch.tensor([s for t, s in t_s], device=self.device)

        # if k is None:
        #     k = config.getx("model/k", 2)
        # self.k = k
        # self.vk = config.getx("model/vk", 128)

    def get_hist(self, users, k):
        assert isinstance(users, torch.Tensor)
        assert len(users.shape) == 1
        assert users.max() < self.t2s.shape[0]
        assert users.min() >= 0
        users = self.t2s[users]
        if k == -1:
            return self.hist_dataset.all_historical_items_on_users(users)
        else:
            return self.hist_dataset.sample_items_on_users(users, k=k)
    #
    # def __getitem__(self, index):
    #     res = super(CFHistAdaptor, self).__getitem__(index)
    #     if len(res) == 3:
    #         users, positives, negatives = res
    #         hists = self.get_hist(users)
    #         return (users, hists), positives, negatives
    #     else:
    #         raise NotImplementedError(f"len(res): {len(res)}")
    #
    # def valid_tensor(self):
    #     users, items = super().valid_tensor()
    #     hists = self.get_hist(users, is_evaluate=True)
    #     return (users, hists), items
    #
    # def tests_tensor(self):
    #     users, items = super().tests_tensor()
    #     hists = self.get_hist(users, is_evaluate=True)
    #     return (users, hists), items
    #
    # def candi_tensor(self):
    #     # users, items = super().candi_tensor()
    #     # assert len(users.shape) == 1
    #     # batch, cand_num = users.reshape([-1, self.candidate_num]).shape
    #     # users = users.reshape([batch, cand_num])[:, 0]
    #     # assert users.shape == torch.Size([batch])
    #     # hists = self.get_hist(users)
    #     # assert hists.shape == torch.Size([batch, self.k])
    #     #
    #     # hists = hists.unsqueeze(1).repeat(1, cand_num, 1).reshape([batch * cand_num, self.k])
    #     #
    #     # return (users, hists), items
    #     raise NotImplementedError
