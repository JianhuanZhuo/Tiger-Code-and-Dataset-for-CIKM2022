import argparse
import os
import queue
import random
from itertools import product
from multiprocessing.managers import BaseManager

# 发送任务的队列:
from tqdm import tqdm

from dworker import multi_worker
from full_trainer import wrap as full_trainer
from tools import config
from tools.config import Config
from tools.config import load_specific_config


def warmup_dataset(exp_config):
    from dataset import dataset_resolver, kgdataset_resolver
    if exp_config.getx(f"dataset/source", None) is not None:
        dataset = dataset_resolver.make(exp_config.getx(f"dataset/source", None),
                                        config=exp_config)
    if exp_config.getx(f"dataset/target", None) is not None:
        dataset = dataset_resolver.make(exp_config.getx(f"dataset/target", None),
                                        config=exp_config, split_mode='all_for_test')
    if exp_config.getx(f"dataset/kg", None) is not None:
        dataset = kgdataset_resolver.make(exp_config.getx(f"dataset/kg", None),
                                          config=exp_config)
    del dataset


def grid_search(gpus, config_file, trainer=full_trainer, setting_filter=None, waits=False):
    def create_worker():
        pid = os.fork()
        if pid != 0:
            multi_worker(gpus, trainer=trainer)

    d_grid_search(config_file, setting_filter=setting_filter, ready_callback=create_worker)


def d_grid_search(config_file, task_process=False, setting_filter=None, ready_callback: callable = None):
    task_queue = queue.Queue()
    feed_queue = queue.Queue()

    server_port = 4135
    server_keys = b'4135'

    # 从BaseManager继承的QueueManager:
    class QueueManager(BaseManager):
        pass

    # 把两个Queue都注册到网络上, callable参数关联了Queue对象:
    QueueManager.register('get_task_queue', callable=lambda: task_queue)
    QueueManager.register('get_feed_queue', callable=lambda: feed_queue)

    if isinstance(config_file, str):
        exp_config = config.load_specific_config(config_file)
    elif isinstance(config_file, Config):
        exp_config = config_file
    else:
        raise NotImplementedError()

    grid = exp_config['_grid_search_']
    repeat = exp_config.getx("_grid_search_repeat", 1)
    exp_config['log_folder'] = 'grid'

    exp_config['log_level'] = "WARNING"
    # warmup_dataset(exp_config)

    dataset_source = exp_config['dataset/source']
    dataset_target = exp_config.getx('dataset/target', 'T')
    dataset_train = exp_config.getx('dataset/third', 'N')
    dataset_KG = exp_config.getx('dataset/kg', 'K')

    dataset_name = f"{dataset_KG}-{dataset_train}={dataset_source}>{dataset_target}"

    model_name = exp_config['model/model']
    time_mask = exp_config["timestamp_mask"]
    if exp_config["git/state"] == "Good":
        time_mask += '-%s' % (exp_config['git']['hexsha'][:5])

    folder = f"{model_name}-{dataset_name}-{time_mask}"

    print(f"output into :{folder}")

    search_space = list(product(*[
        vs if isinstance(vs, list) else eval(f'{vs}')
        for vs in grid.values()
    ]))
    random.shuffle(search_space)

    if setting_filter is not None:
        search_space = [
            setting
            for setting in search_space
            if setting_filter({
                k: v
                for k, v in zip(list(grid.keys()), setting)
            })
        ]

    total = repeat * len(search_space)
    # assert total <= len(gpus), f"total <= len(gpus): total {total} : len(gpus) {len(gpus)}"
    exp_config['grid_spec/total'] = total

    # 绑定端口5000, 设置验证码'mima123':
    with QueueManager(address=('', server_port), authkey=server_keys) as manager:
        # 启动Queue:
        # manager.start()
        task = manager.get_task_queue()
        feed = manager.get_feed_queue()

        task_num = 0
        for _ in tqdm(range(repeat), desc='load task'):
            for i, setting in enumerate(search_space):
                # print(setting)
                for idx, k in enumerate(grid.keys()):
                    exp_config[k] = setting[idx]
                exp_config['grid_spec/current'] = task_num
                task.put(exp_config.clone())
                task_num += 1
            exp_config.random_again()

        if ready_callback is not None:
            ready_callback()

        bar = tqdm(range(task_num), desc='tasks') if task_process else range(task_num)
        for _ in bar:
            s = feed.get()
            if task_process:
                bar.postfix = f"{s}"


def f(setting):
    if 'model/layer_slice' in setting and 'model/n_layers' in setting:
        return setting['model/layer_slice'] <= setting['model/n_layers']
    return True


def exp_pack(config_file, trainer=full_trainer):
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['work', 'serve', 'run'],
                        nargs='?',
                        default='run',
                        help="mode in ['work', 'serve', 'run']")
    parser.add_argument('--tpg', default=1, required=False, type=int, help="task per gpu")

    args = parser.parse_args()
    if args.mode == 'run':
        cfg = load_specific_config(config_file)
        cfg['train/print_eval'] = True
        cfg['train/print_best'] = True
        cfg['train/epoch_tqdm'] = False
        cfg['train/batch_tqdm'] = True
        trainer(cfg)
    elif args.mode == 'serve':
        cfg = load_specific_config(config_file)
        cfg['log_level'] = "WARMING"
        d_grid_search(cfg, task_process=True, setting_filter=f)
    elif args.mode == 'work':
        t = args.tpg
        import GPUtil
        gpu_list = list(range(len(GPUtil.getGPUs())))
        multi_worker(gpu_list * t, trainer=trainer)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    print("please use the dl-search or rd-search!")
