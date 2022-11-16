import dgl
import dgl.function as fn
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, Embedding

from tools.utils import Timer


class GCNLayer(nn.Module):
    def __init__(self, graph, norm):
        super(GCNLayer, self).__init__()
        self.graph = graph
        self.norm = norm

    def forward(self, node_f):
        with self.graph.local_scope():
            node_f = node_f * self.norm

            self.graph.ndata['n_f'] = node_f
            self.graph.update_all(message_func=fn.copy_src(src='n_f', out='m'), reduce_func=fn.sum(msg='m', out='n_f'))

            rst = self.graph.ndata['n_f']

            rst = rst * self.norm

            return rst


class LGNRecModel(Module):
    def __init__(self, config, kg_dataset):
        super().__init__()
        self.config = config
        self.device = config.getx("device", "cuda")
        self.num_user = config['num_user']
        self.dim = config['model/dim']

        self.n_layers = self.config['model/n_layers']
        self.drop = self.config['model/drop']
        self.decay = self.config['model/decay']
        self.layer_slice = self.config.getx("model/layer_slice", 0)

        hs = torch.tensor(kg_dataset.kg[0], device=self.device)
        ts = torch.tensor(kg_dataset.kg[1], device=self.device)
        with Timer("Construct G"):
            g = self.inter_graph = dgl.graph((hs, ts),
                                             num_nodes=kg_dataset.num_entity,
                                             idtype=torch.int32,
                                             device=self.device)
        self.entity_embedding = Embedding(kg_dataset.num_entity, self.dim)
        self.user_embedding = Embedding(self.num_user, self.dim)

        with torch.no_grad():
            d_ins = g.in_degrees()
            d_out = g.out_degrees()
            norm = (d_ins + d_out).float().clamp(min=1).pow(-0.5).view(-1, 1)

        self.layers = nn.ModuleList([GCNLayer(self.inter_graph, norm) for _ in range(self.n_layers)])

    def compute_graph(self):
        """
        propagate methods for GCN
        """
        all_emb = self.entity_embedding.weight

        gcn_out = torch.zeros_like(all_emb)
        for i, layer in enumerate(self.layers):
            if self.layer_slice <= i:
                gcn_out += all_emb
            all_emb = layer(all_emb)
        gcn_out += all_emb

        gcn_out = gcn_out / (self.n_layers - self.layer_slice + 1)

        return gcn_out

    def forward(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        assert users.shape == pos_items.shape == neg_items.shape

        item_embedding = self.compute_graph()
        pos_i_emb = item_embedding[pos_items]
        neg_i_emb = item_embedding[neg_items]

        users_emb = self.user_embedding(users)

        pos_i_emb = F.dropout(pos_i_emb, p=self.drop)
        neg_i_emb = F.dropout(neg_i_emb, p=self.drop)
        users_emb = F.dropout(users_emb, p=self.drop)

        pos_scores = torch.sum(users_emb * pos_i_emb, dim=-1)
        neg_scores = torch.sum(users_emb * neg_i_emb, dim=-1)

        assert pos_scores.shape == torch.Size([batch_size])
        assert neg_scores.shape == torch.Size([batch_size])

        loss = F.softplus(neg_scores - pos_scores)

        return loss

    def part_distances(self, users, items):
        with torch.no_grad():
            users_emb = self.user_embedding(users)
            entity_embedding = self.compute_graph()
            items_emb = entity_embedding[items]
            rating = torch.sum(users_emb * items_emb, -1)
            return rating

    def full_user_scores(self, users, full_items=None):
        assert len(full_items.shape) == 1
        # assert len(users.shape) == 1
        item_num = full_items.shape[0]
        batch_size = users[0].shape[0] if isinstance(users, tuple) else users.shape[0]
        with torch.no_grad():
            entity_embedding = self.compute_graph()
            users_emb = self.user_embedding(users)
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
