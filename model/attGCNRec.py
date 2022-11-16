import torch
import torch.nn.functional as F

from model import AvgLGNRecModel


class AttLGNRecModel(AvgLGNRecModel):
    def __init__(self, config, kg_dataset, compute_graph=None):
        super().__init__(config, kg_dataset, compute_graph=compute_graph)
        self.gate_nn = torch.nn.Linear(self.dim, 1)

    def get_user_embedding(self, hists, item_embeds):
        if isinstance(hists, (list, tuple)):
            assert len(hists) == 2
            users, hists = hists
        assert len(hists.shape) == 2

        batch_size, hist_len = hists.shape
        assert hist_len == self.k
        hist_embeds = item_embeds[hists]
        assert hist_embeds.shape == torch.Size([batch_size, hist_len, self.dim])
        hist_scores = self.gate_nn(hist_embeds).reshape([batch_size, 1, hist_len])
        hist_scores = F.softmax(hist_scores, dim=2)
        out_put = torch.reshape(hist_scores @ hist_embeds, [batch_size, self.dim])
        return out_put
