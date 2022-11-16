import json
import os

import torch
import numpy as np


class FullEvaluator:
    def __init__(self, config, summary, dataset, prefix=""):
        self.config = config
        self.summary = summary
        self.dataset = dataset
        self.device = config.getx("device", "cuda")
        self.prefix = prefix
        self.stop = False

        self.size_valid = len(dataset.valid)

        self.stop_delay = config.get_or_default('evaluator_args/stop_delay', 10)
        self.score_cache = [-np.inf]
        self.best_test_performance = []

        self.pack_user = torch.tensor([u for u, i in dataset.valid + dataset.tests], device=self.device)
        self.pack_item = torch.tensor([i for u, i in dataset.valid + dataset.tests], device=self.device)
        assert self.pack_item.shape == self.pack_user.shape
        assert len(self.pack_item.shape) == 1
        self.pack_size = self.pack_item.shape[0]
        assert self.pack_size == self.size_valid * 2

        self.full_items = torch.arange(self.dataset.num_item, device=self.device)
        self.full_items_size = self.full_items.shape[0]

        self.unique_users, self.inverse_indices = torch.unique(self.pack_user, return_inverse=True)
        self.unique_users_size = self.unique_users.shape[0]

        full_uis = sorted(self.dataset.train + self.dataset.tests + self.dataset.valid, key=lambda x: x[0])
        self.real_user, self.real_items = torch.tensor(full_uis).split([1, 1], dim=1)

    def reset(self):
        self.stop = False
        self.score_cache = [-np.inf]

    def evaluate(self, model, epoch):
        if self.stop:
            return
        # 失能验证
        if self.config.getx("eval_disable", False):
            return
        model.eval()
        with torch.no_grad():
            xs = self.config.getx("evaluator_args/eval_xs", [10, 100])
            valid_rank, tests_rank = self.full_rank_position(model)
            this_is_best = False
            valid_ndcg = (1 / torch.log2((valid_rank[valid_rank < 100] + 2))).sum().item() / self.size_valid * 100
            if valid_ndcg > max(self.score_cache):
                self.score_cache = [valid_ndcg]
                this_is_best = True
            else:
                self.score_cache.append(valid_ndcg)

            ranks = tests_rank
            assert ranks.shape == torch.Size([self.size_valid])
            # assert ranks.min().item() == 0
            # assert ranks.max().item() <= 100
            evals = [f"{valid_ndcg:5.4f}"]

            if self.summary:
                self.summary.add_scalar(f"{self.prefix}valid/ndcg@10", valid_ndcg, global_step=epoch)

            test_performance = []

            for x in xs:
                hitx = (ranks < x).float().mean().item() * 100
                # 加2因为，这里的排名是 从0 开始的，不是 1
                ndcg = (1 / torch.log2((ranks[ranks < x] + 2))).sum().item() / self.size_valid * 100
                MRRx = (1 / (ranks[ranks < x] + 1)).sum().item() / self.size_valid * 100

                test_performance.append((f"hit@{x}", hitx))
                if x != 1:
                    test_performance.append((f"ndcg@{x}", ndcg))
                    test_performance.append((f"MRR@{x}", MRRx))

                # only display @5 and @10 during training
                if x == 1:
                    evals.append(f"{x}: {hitx:5.4f}")
                # elif x == 10:
                #     evals.append(f"{x}: {hitx:5.4f}/{ndcg:5.4f}/{MRRx:5.4f}")
                else:
                    evals.append(f"{x}: {hitx:5.4f}/{ndcg:5.4f}/{MRRx:5.4f}")

            if this_is_best:
                evals.append(f"Best")
                self.best_test_performance = test_performance
            else:
                evals.append(str(len(self.score_cache)))

            if self.config.getx("train/print_eval", False):
                print(f" {self.prefix}Eval {epoch:4} | {' | '.join(evals)}", flush=True)

            if self.summary:
                for label, value in test_performance:
                    self.summary.add_scalar(f"{self.prefix}eval/" + label, value, global_step=epoch)
            return this_is_best

    def full_rank_position(self, model):
        # get full scores of unique users
        unique_users_scores = model.full_user_scores(self.unique_users, self.full_items, dataset=self.dataset)
        assert unique_users_scores.shape == torch.Size([self.unique_users_size, self.full_items_size])

        # get scores of target item
        pack_item_scores = unique_users_scores[self.inverse_indices, self.pack_item]
        assert pack_item_scores.shape == torch.Size([self.pack_size])

        # mask scores of all positive sample
        assert self.real_user.max() < self.unique_users_size
        assert self.real_items.max() < self.full_items_size
        unique_users_scores[self.real_user, self.real_items] = - np.inf

        # full masked score for each user in pack
        pack_user_score_full_mask = unique_users_scores[self.inverse_indices]
        assert pack_user_score_full_mask.shape == torch.Size([self.pack_size, self.full_items_size])

        # un-squeeze for each pack item
        pack_item_scores = pack_item_scores.unsqueeze(1)
        assert pack_item_scores.shape == torch.Size([self.pack_size, 1])

        # ge to get the rank of each pack item in the full score of user
        # pack_ranks = torch.ge(pack_user_score_full_mask, pack_item_scores).sum(dim=-1)
        user_size = pack_user_score_full_mask.shape[0]
        small_size = 1000
        pack_ranks = []
        for b in range(user_size // small_size + 1):
            s = b * small_size
            e = (b + 1) * small_size
            pack_ranks.append(torch.ge(pack_user_score_full_mask[s:e], pack_item_scores[s:e]).sum(dim=-1))
        pack_ranks = torch.cat(pack_ranks)

        assert pack_ranks.shape == torch.Size([self.pack_size])

        valid_rank = pack_ranks[:self.size_valid]
        tests_rank = pack_ranks[self.size_valid:]

        return valid_rank, tests_rank

    def should_stop(self):
        if self.stop:
            return True
        if self.config.get_or_default("evaluator_args/use_stop", False):
            self.stop = len(self.score_cache) >= self.stop_delay
            return self.stop
        return False

    def record_best(self, epoch_run):
        print_str = []
        for label, value in self.best_test_performance:
            if self.summary:
                self.summary.add_scalar(f"{self.prefix}best/" + label, value, global_step=0)
            print_str.append(f"{value:5.4f}")
        if self.config.getx("train/print_best", False):
            _grid_thread_hit1 = self.config.getx("_grid_thread_hit1", None)
            if "grid_spec" in self.config and _grid_thread_hit1 is not None:
                hit1 = self.best_test_performance[0][1]
                if hit1 < _grid_thread_hit1:
                    return
            s = '|'.join(print_str)
            print(f" {self.prefix}Best {s}")

        if not os.path.exists('../output'):
            os.makedirs('../output', exist_ok=True)

        dataset_source = self.config['dataset/source']
        dataset_target = self.config.getx('dataset/target', 'T')
        dataset_train = self.config.getx('dataset/third', 'N')
        dataset_KG = self.config.getx('dataset/kg', 'K')

        dataset_name = f"{dataset_KG}-{dataset_train}={dataset_source}>{dataset_target}"

        time_mask = self.config["timestamp_mask"]
        if self.config["git/state"] == "Good":
            time_mask += '-%s' % (self.config['git']['hexsha'][:5])

        model_name = self.config['model/model']

        sub_folder = f"{model_name}-{dataset_name}-{time_mask}"

        record_dir = os.path.join('../output', sub_folder)
        record_file = self.prefix + self.config.postfix() + '.json'
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)
        with open(os.path.join(record_dir, record_file), 'w') as fp:
            json.dump({
                'config': self.config,
                'best': self.best_test_performance,
                'epoch_run': epoch_run,
            }, fp)


if __name__ == '__main__':
    pass
