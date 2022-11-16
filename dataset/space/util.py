import torch

from tools.utils import group_kv


def u2is(dataset, device='cuda'):
    u2is_list = group_kv(dataset, return_type='list')
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

    train_uisl_full = torch.tensor(full_u2is, device=device)
    ux_start_full = torch.tensor(ux_start, device=device).reshape([-1, 1])
    ux_len_full = torch.tensor(ux_len, device=device).reshape([-1, 1])

    def func(u):
        s = ux_start_full[u]
        e = s + ux_len_full[u]
        return train_uisl_full[s:e]

    return func
