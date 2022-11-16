import torch
from torch.nn import functional as F, Dropout

from model import BasedModel


class SmBPRModel(BasedModel):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.drop = Dropout(self.config.getx("model/drop", 0))
        self.decay = self.config['model/decay']
        self.sections = self.config.getx("model/sections", [2, 16])
        assert self.dim == self.sections[0] * self.sections[1]

    def score(self, u_emb, i_emb):
        assert u_emb.shape == i_emb.shape
        assert i_emb.shape[-1] == self.dim
        m = list(u_emb.shape[:-1])
        af_m = m + self.sections
        u_emb = u_emb.reshape(af_m)
        i_emb = i_emb.reshape(af_m)
        scores = torch.sum(u_emb * i_emb, dim=-1)
        # sm_score = torch.softmax(raw_score, dim=-1)
        # scores = torch.log_softmax(scores, dim=-1)
        res = scores.max(dim=-1)[0]
        assert res.shape == torch.Size(m)
        return res

    def forward(self, user, positive, negative, nbs=None, pos_nbs=None, neg_nbs=None):
        assert nbs is None and pos_nbs is None and neg_nbs is None
        batch_size = user.shape[0]
        assert user.shape == positive.shape == negative.shape == torch.Size([batch_size])
        u_emb = self.drop(self.user_embedding(user))
        p_emb = self.drop(self.item_embedding(positive))
        n_emb = self.drop(self.item_embedding(negative))

        reg_loss = torch.cat([u_emb, p_emb, n_emb]).norm(2).pow(2) / float(batch_size) / 2

        # up_score = torch.sum(u_emb * p_emb, dim=-1)
        # un_score = torch.sum(u_emb * n_emb, dim=-1)
        up_score = self.score(u_emb, p_emb)
        un_score = self.score(u_emb, n_emb)
        assert up_score.shape == torch.Size([batch_size])
        assert un_score.shape == torch.Size([batch_size])

        return F.softplus(un_score - up_score) + reg_loss * self.decay

    def part_distances(self, users, items, nbs=None, item_nbs=None):
        assert nbs is None and item_nbs is None
        with torch.no_grad():
            user_embed = self.user_embedding(users)
            item_embed = self.item_embedding(items)
            # return torch.sum(user_embed * item_embed, dim=-1)
            return self.score(user_embed, item_embed)

    def additional_regularization(self):
        return 0

    def epoch_hook(self, epoch):
        pass

    def batch_hook(self, epoch, batch):
        pass
