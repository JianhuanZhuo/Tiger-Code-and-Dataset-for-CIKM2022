import json
import os

import torch
import numpy as np


class Evaluator:

    def __init__(self, config, summary, dataset, prefix="", i2k_mapper=None):
        self.config = config
        self.summary = summary
        self.dataset = dataset
        self.device = config.getx("device", "cuda")
        self.candidate_num = config.getx("dataset/candidate_num", 100)
        self.prefix = prefix
        self.i2k_mapper = i2k_mapper
        self.stop = False

        self.size_valid = len(dataset.valid)

        self.low_first = config.get_or_default('evaluator_args/low_first', True)

        self.pack_user, self.pack_item = dataset.pack_tensor()

        self.stop_delay = config.get_or_default('evaluator_args/stop_delay', 10)
        self.score_cache = [-np.inf]
        self.best_test_performance = []

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
            xs = self.config.getx("evaluator_args/eval_xs", [1, 10])

            scores = model.part_distances(self.pack_user, self.pack_item)

            valid_rank, tests_rank = self.rank_position(scores)
            this_is_best = False
            valid_ndcg = (1 / torch.log2((valid_rank[valid_rank < 10] + 2))).sum().item() / self.size_valid
            if valid_ndcg > max(self.score_cache) + 0.001:
                self.score_cache = [valid_ndcg]
                this_is_best = True
            else:
                self.score_cache.append(valid_ndcg)

            ranks = tests_rank
            assert ranks.shape == torch.Size([self.size_valid])
            # assert ranks.min().item() == 0
            assert ranks.max().item() <= 100
            evals = [f"{valid_ndcg:5.4f}"]

            if self.summary:
                self.summary.add_scalar(f"{self.prefix}valid/ndcg@10", valid_ndcg, global_step=epoch)

            test_performance = []

            for x in xs:
                hitx = (ranks < x).float().mean().item()
                # 加2因为，这里的排名是 从0 开始的，不是 1
                ndcg = (1 / torch.log2((ranks[ranks < x] + 2))).sum().item() / self.size_valid
                MRRx = (1 / (ranks[ranks < x] + 1)).sum().item() / self.size_valid

                test_performance.append((f"hit@{x}", hitx))
                if x != 1:
                    test_performance.append((f"ndcg@{x}", ndcg))
                    test_performance.append((f"MRR@{x}", MRRx))

                # only display @5 and @10 during training
                if x == 1:
                    evals.append(f"{x}: {hitx:5.4f}")
                elif x == 10:
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

    def rank_position(self, distances):
        assert distances.shape == torch.Size([self.size_valid * 102])

        valid = distances[:self.size_valid].reshape([self.size_valid, 1])
        tests = distances[self.size_valid:self.size_valid * 2].reshape([self.size_valid, 1])
        candi = distances[self.size_valid * 2:].reshape([self.size_valid, 100])

        # the number of topK of negative that bigger than positive is the rank of positive
        # Large is good
        valid_rank = (candi >= valid).sum(dim=-1)
        tests_rank = (candi >= tests).sum(dim=-1)

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
        s = '|'.join(print_str)
        if self.config.getx("train/print_best", False):
            _grid_thread_hit1 = self.config.getx("_grid_thread_hit1", None)
            if "grid_spec" in self.config and _grid_thread_hit1 is not None:
                hit1 = self.best_test_performance[0][1]
                if hit1 < _grid_thread_hit1:
                    return
            print(f" {self.prefix}Best {s}")
        if not os.path.exists('../output'):
            os.makedirs('../output', exist_ok=True)

        dataset_source = self.config['dataset/source'][:5]
        dataset_target = self.config.getx('dataset/target', 'T')[:5]
        dataset_train = self.config.getx('dataset/third', 'N')[:5]
        dataset_KG = self.config.getx('dataset/kg', 'K')[:5]

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
