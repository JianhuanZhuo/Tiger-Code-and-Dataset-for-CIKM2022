from abc import ABC

import torch
from torch import nn
from torch.nn import Module, Embedding, Parameter, Dropout


class ENMFLoss(Module, ABC):
    def __init__(self, user_num, item_num, config):
        super().__init__()
        self.config = config
        self.item_num = item_num
        self.dim = config['model/dim']
        self.drop_out = config.getx("model/drop", 0.3)
        self.neg_weight = config.getx("model/neg_weight", 0.1)

        self.user_embedding = Embedding(user_num, self.dim)
        self.item_embedding = Embedding(item_num + 1, self.dim)
        self.h = Parameter(torch.randn(self.dim, 1))
        self.dropout = Dropout(self.drop_out)
        self.max_train_len = -1

        def shape_check(module, fea_in, fea_out):
            user, positive = fea_in
            batch_size = user.shape[0]
            assert user.shape == torch.Size([batch_size])
            if self.max_train_len == -1:
                self.max_train_len = positive.shape[1]
            else:
                assert positive.shape == torch.Size([batch_size, self.max_train_len])

            assert fea_out.shape == torch.Size([batch_size])

            return None

        self.register_forward_hook(hook=shape_check)

        self.reset_para()

    def reset_para(self):
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        nn.init.constant_(self.h, 0.01)

    # def rank(self, uid):
    #     '''
    #     uid: Batch_size
    #     '''
    #     uid_embs = self.user_embs(uid)
    #     user_all_items = uid_embs.unsqueeze(1) * self.item_embs.weight
    #     items_score = user_all_items.matmul(self.h).squeeze(2)
    #     return items_score

    def part_distances(self, users, items):
        with torch.no_grad():
            user_embed = self.user_embedding(users)
            item_embed = self.item_embedding(items)

            user_items = user_embed * item_embed
            assert len(user_items.shape) == 2
            items_score = user_items.matmul(self.h).squeeze(-1)
            return - items_score

    def additional_regularization(self):
        return 0

    def epoch_hook(self, epoch):
        pass

    def batch_hook(self, epoch, batch):
        pass

    def forward(self, uids, pos_iids):
        '''
        uids: B
        u_iids: B * L
        '''
        batch_size = uids.shape[0]
        if self.max_train_len == -1:
            self.max_train_len = pos_iids.shape[1]

        u_emb = self.dropout(self.user_embedding(uids))
        pos_embs = self.item_embedding(pos_iids)

        # torch.einsum("ab,abc->abc")
        # B * L * D
        mask = (~(pos_iids.eq(self.item_num))).float()
        pos_embs = pos_embs * mask.unsqueeze(2)

        # torch.einsum("ac,abc->abc")
        # B * L * D
        pq = u_emb.unsqueeze(1) * pos_embs
        # torch.einsum("ajk,kl->ajl")
        # B * L
        hpq = pq.matmul(self.h).squeeze(2)

        # loss
        pos_data_loss = torch.sum((1 - self.neg_weight) * hpq.square() - 2.0 * hpq, dim=1)

        assert pos_data_loss.shape == torch.Size([batch_size])

        # torch.einsum("ab,ac->abc")
        part_1 = self.item_embedding.weight.unsqueeze(2).bmm(self.item_embedding.weight.unsqueeze(1))
        part_2 = u_emb.unsqueeze(2).bmm(u_emb.unsqueeze(1))

        # D * D
        part_1 = part_1.sum(0)
        part_2 = part_2.sum(0)
        part_3 = self.h.mm(self.h.t())
        all_data_loss = torch.sum(part_1 * part_2 * part_3) / batch_size

        loss = self.neg_weight * all_data_loss + pos_data_loss
        return loss
