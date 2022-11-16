from torch.nn import Embedding, Module
import torch
from torch.nn import functional as F, Dropout

from model.GCN import GCNBase


class GCNEmb(Module):
    def __init__(self, config, kg_dataset):
        super().__init__()
        self.config = config
        self.dim = config['model/dim']
        self.e_num = config['num_entity']
        self.r_num = config['num_relation']
        # self.entity_embedding = Embedding(self.e_num, self.dim)
        # self.relation_embedding = Embedding(self.r_num, self.dim)
        self.drop = Dropout(config.getx("model/drop", 0))
        self.compute_graph = GCNBase(config, kg_dataset)

    def forward(self, *args, mode='loss'):
        if mode == 'loss':
            return self.get_loss(*args)
        # elif mode == 'part_scores':
        #     return self.part_scores(*args)
        else:
            raise NotImplementedError(f"not known mode:{mode}")


    def get_loss(self, h, r, tp, tn):
        assert h.shape == r.shape == tp.shape == tn.shape
        batch_size = h.shape[0]

        entity_embeddings = self.compute_graph()

        h_e = entity_embeddings[h]
        tp_e = entity_embeddings[tp]
        tn_e = entity_embeddings[tn]

        up_score = ((h_e - tp_e) ** 2).sum(dim=1)
        un_score = ((h_e - tn_e) ** 2).sum(dim=1)

        # h_e = self.drop(self.entity_embedding(h))
        # r_e = self.drop(self.relation_embedding(r))
        # tp_e = self.drop(self.entity_embedding(tp))
        # tn_e = self.drop(self.entity_embedding(tn))
        #
        # up_score = ((h_e + r_e - tp_e) ** 2).sum(dim=1)
        # un_score = ((h_e + r_e - tn_e) ** 2).sum(dim=1)

        assert up_score.shape == un_score.shape == torch.Size([batch_size])
        # BPR loss
        return F.softplus(un_score - up_score)

    # @torch.no_grad()
    # def part_scores(self, hs, rs, tps, tns):
    #     assert hs.shape == rs.shape == tps.shape
    #     assert len(hs.shape) == 2
    #     assert hs.shape[1] == 2
    #     assert len(tns.shape) == 2
    #     assert tns.shape[0] == hs.shape[0]
    #     batch_size, cand_num = tns.shape
    #
    #     h_e = self.entity_embedding(hs)
    #     r_e = self.relation_embedding(rs)
    #     tp_e = self.entity_embedding(tps)
    #     tns_e = self.entity_embedding(tns)
    #
    #     hr = h_e + r_e
    #     assert tp_e.shape == hr.shape == torch.Size([batch_size, 2, self.dim])
    #     assert tns_e.shape == torch.Size([batch_size, cand_num, self.dim])
    #
    #     p_score = ((hr - tp_e) ** 2).sum(dim=2)
    #     assert p_score.shape == torch.Size([batch_size, 2])
    #     p_score = p_score.unsqueeze(1)
    #
    #     hr = hr.unsqueeze(1)
    #     assert hr.shape == torch.Size([batch_size, 1, 2, self.dim])
    #     tns_e = tns_e.unsqueeze(2)
    #     assert tns_e.shape == torch.Size([batch_size, cand_num, 1, self.dim])
    #     ns_score = ((hr - tns_e) ** 2).sum(dim=3)
    #     assert ns_score.shape == torch.Size([batch_size, cand_num, 2])
    #
    #     res = torch.cat([p_score, ns_score], dim=1)
    #     assert res.shape == torch.Size([batch_size, cand_num+1, 2])
    #     return res

    def epoch_hook(self, epoch):
        # assert self.entity_embedding.weight.isnan().sum() == 0
        # assert self.relation_embedding.weight.isnan().sum() == 0
        pass

    def batch_hook(self, epoch, batch):
        pass

    def additional_regularization(self):
        return 0
