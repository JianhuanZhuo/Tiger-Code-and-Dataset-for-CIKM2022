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


class GCNBase(Module):
    def __init__(self, config, kg_dataset):
        super().__init__()
        self.config = config
        self.device = config.getx("device", "cuda")
        self.num_user = config['num_user']
        self.dim = config['model/dim']

        self.n_layers = self.config['model/n_layers']
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

        with torch.no_grad():
            d_ins = g.in_degrees()
            d_out = g.out_degrees()
            norm = (d_ins + d_out).float().clamp(min=1).pow(-0.5).view(-1, 1)

        self.layers = nn.ModuleList([GCNLayer(self.inter_graph, norm) for _ in range(self.n_layers)])
        self.k = self.config.getx("model/k", 8)

    def forward(self):
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
