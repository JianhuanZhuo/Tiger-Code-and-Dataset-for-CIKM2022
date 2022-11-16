import torch
from torch.nn import functional as F, Dropout, EmbeddingBag

from model import BPRModel, BasedModel


class UserHisAvgModel(BasedModel):
    def __init__(self, config, dataset, history_dataset):
        super().__init__(config, dataset)
        self.history_dataset = history_dataset
        self.dataset = dataset
        assert history_dataset.num_user == dataset.num_user
        self.user_embedding_bag = EmbeddingBag(
            num_embeddings=history_dataset.num_item,
            embedding_dim=32,
            mode='mean'
        )

    def get_user_hist(self, u):
        if isinstance(u, torch.Tensor):
            return list(self.history_dataset.full_u2is[u.item()])
        elif isinstance(u, int):
            return list(self.history_dataset.full_u2is[u])
        else:
            raise NotImplementedError(type(u))

    def get_embedding(self, users):
        batch_size = users.shape[0]
        history_arr = []
        history_off = []
        for u in users:
            history_off.append(len(history_arr))
            history_arr += self.get_user_hist(u)
        bag = torch.tensor(history_arr, device=self.device)
        off = torch.tensor(history_off, device=self.device)
        res = self.user_embedding_bag(bag, off)
        assert res.shape == torch.Size([batch_size, self.dim])
        return res

    def forward(self, user, positive, negative):
        assert user.shape == positive.shape == negative.shape
        batch_size = user.shape[0]
        u_emb = self.get_embedding(user)
        p_emb = self.item_embedding(positive)
        n_emb = self.item_embedding(negative)

        up_score = torch.sum(u_emb * p_emb, dim=-1)
        un_score = torch.sum(u_emb * n_emb, dim=-1)
        assert up_score.shape == torch.Size([batch_size])
        assert un_score.shape == torch.Size([batch_size])

        return F.softplus(un_score - up_score)

    def part_distances(self, users, items, nbs=None, item_nbs=None):
        assert nbs is None and item_nbs is None
        with torch.no_grad():
            user_embed = self.get_embedding(users)
            item_embed = self.item_embedding(items)
            return torch.sum(user_embed * item_embed, dim=-1)

    def additional_regularization(self):
        return 0

    def epoch_hook(self, epoch):
        pass

    def batch_hook(self, epoch, batch):
        pass
