import torch
import torch.nn.functional as F
from torch.nn import Module

from model.GCN import GCNBase
from dataset.cf_kg_adaptor import CFKGAdaptor
from dataset.cf_hist_avg_adaptor import CFHistAdaptor

class AvgLGNRecModel(Module):
    def __init__(self, config,
                 kg_dataset,
                 compute_graph=None,
                 ):
        super().__init__()
        self.config = config
        self.device = config.getx("device", "cuda")
        self.num_user = config['num_user']
        self.dim = config['model/dim']
        self.k = self.config.getx("model/k", 8)
        self.vk = self.config.getx("model/k", 128)
        self.drop = self.config['model/drop']
        self.decay = self.config['model/decay']
        if compute_graph is None:
            compute_graph = GCNBase(config, kg_dataset)
        self.compute_graph = compute_graph

    def get_user_embedding(self, hists, item_embeds):
        if isinstance(hists, torch.Tensor):
            assert len(hists.shape) == 2
            return F.embedding_bag(hists, item_embeds)
        elif isinstance(hists, (tuple, list)):
            assert len(hists) == 2
            hists, offset = hists
            assert len(hists.shape) == 1
            return F.embedding_bag(hists, weight=item_embeds, offsets=offset)
        else:
            raise NotImplementedError

    def user_convert(self, users, dataset, is_evaluate=False):
        assert dataset is not None
        assert isinstance(dataset, CFHistAdaptor)
        users = dataset.get_hist(users, k=self.vk if is_evaluate else self.k)
        assert isinstance(dataset, CFKGAdaptor)
        if isinstance(users, (tuple, list)):
            hist, lens = users
            assert hasattr(dataset, 'hist_dataset')
            hist = dataset.hist_dataset.i2e(hist)
            users = hist, lens
        return users

    def item_convert(self, items, dataset):
        assert dataset is not None
        assert isinstance(dataset, CFKGAdaptor)
        items = dataset.i2e(items)
        return items

    def forward(self, users, pos_items, neg_items, dataset=None, is_evaluate=False):
        users = self.user_convert(users, dataset, is_evaluate=is_evaluate)
        pos_items = self.item_convert(pos_items, dataset=dataset)
        neg_items = self.item_convert(neg_items, dataset=dataset)

        return self.scores(users, pos_items, neg_items)

    def scores(self, users, pos_items, neg_items):
        batch_size = pos_items.shape[0]
        # assert users.shape == pos_items.shape == neg_items.shape

        entity_embedding = self.compute_graph()
        pos_i_emb = entity_embedding[pos_items]
        neg_i_emb = entity_embedding[neg_items]

        users_emb = self.get_user_embedding(users, entity_embedding)

        pos_i_emb = F.dropout(pos_i_emb, p=self.drop)
        neg_i_emb = F.dropout(neg_i_emb, p=self.drop)
        users_emb = F.dropout(users_emb, p=self.drop)

        reg_loss = torch.cat([users_emb, pos_i_emb, neg_i_emb]).norm(2).pow(2) / float(batch_size) / 2

        pos_scores = torch.sum(users_emb * pos_i_emb, dim=-1)
        neg_scores = torch.sum(users_emb * neg_i_emb, dim=-1)

        assert pos_scores.shape == torch.Size([batch_size])
        assert neg_scores.shape == torch.Size([batch_size])

        loss = F.softplus(neg_scores - pos_scores) + reg_loss * self.decay

        return loss

    def part_distances(self, users, items):
        # with torch.no_grad():
        #     entity_embedding = self.compute_graph()
        #     users_emb = self.get_user_embedding(users, entity_embedding)
        #     items_emb = entity_embedding[items]
        #     rating = torch.sum(users_emb * items_emb, -1)
        #     return rating
        raise NotImplementedError

    def full_user_scores(self, users, full_items=None, dataset=None):
        batch_size = users.shape[0]
        users = self.user_convert(users, dataset, is_evaluate=True)
        full_items = self.item_convert(full_items, dataset=dataset)

        assert len(full_items.shape) == 1
        # assert len(users.shape) == 1
        item_num = full_items.shape[0]
        with torch.no_grad():
            entity_embedding = self.compute_graph()
            users_emb = self.get_user_embedding(users, entity_embedding)
            assert users_emb.shape == torch.Size([batch_size, self.dim])

            full_items_emb = entity_embedding[full_items].T
            assert full_items_emb.shape == torch.Size([self.dim, item_num])

            full_user_scores = users_emb @ full_items_emb
            assert full_user_scores.shape == torch.Size([batch_size, item_num])
            return full_user_scores

    def additional_regularization(self):
        return 0

    def epoch_hook(self, epoch):
        pass

    def batch_hook(self, epoch, batch):
        pass
