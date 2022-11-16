import torch
from torch import Tensor

from dataset import ML100KDataset


class CFKGAdaptor(ML100KDataset):
    def __init__(self, config, kg_e_map=None, **kwargs):
        super().__init__(config, **kwargs)
        assert kg_e_map is not None
        self.i_map = self.i_map
        self.e_map = kg_e_map

        self.item_2_kg = sorted([
            (idx, self.e_map[x])
            for x, idx in self.i_map.items()
        ], key=lambda x: x[0])

        assert all([i == ik[0] for i, ik in enumerate(self.item_2_kg)])

        self._i2e = torch.tensor([e_id for i_id, e_id in self.item_2_kg], device=self.device)

    def i2e(self, items):
        assert isinstance(items, torch.Tensor)
        assert items.max() < len(self.i_map)
        assert items.min() >= 0
        return self._i2e[items]

    # def __getitem__(self, index):
    #     res = super(CFKGAdaptor, self).__getitem__(index)
    #     if len(res) == 3:
    #         users, positives, negatives = res
    #         assert isinstance(positives, Tensor)
    #         assert isinstance(negatives, Tensor)
    #
    #         positives = self._i2e[positives]
    #         negatives = self._i2e[negatives]
    #         return users, positives, negatives
    #     else:
    #         raise NotImplementedError(f"len(res): {len(res)}")
    #
    # def get_full_hist_by_user(self, u):
    #     res = super().get_full_hist_by_user(u)
    #     return self._i2e[res]
    #
    # def full_items(self):
    #     res = super().full_items()
    #     return self._i2e[res]
    #
    # def valid_tensor(self):
    #     vu, vi = super().valid_tensor()
    #     vi = self._i2e[vi]
    #     return vu, vi
    #
    # def tests_tensor(self):
    #     tu, ti = super().tests_tensor()
    #     ti = self._i2e[ti]
    #     return tu, ti
    #
    # def candi_tensor(self):
    #     cu, ci = super().candi_tensor()
    #     ci = self._i2e[ci]
    #     return cu, ci
    #
    # def sample_items_on_users(self, users, k):
    #     res = super().sample_items_on_users(users, k)
    #     return self._i2e[res]
