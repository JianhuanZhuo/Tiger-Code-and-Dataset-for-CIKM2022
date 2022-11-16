import logging
import os
import random
import sys
import traceback

import torch
from setproctitle import setproctitle
from torch.optim.adagrad import Adagrad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from dataset import dataset_resolver, kgdataset_resolver
from full_evaluator import FullEvaluator
from model import op_resolver, model_resolver
from tools.tee import StdoutTee, StderrTee
from tools.utils import Timer, timer

logger = logging.getLogger()


def wrap(config):
    logging.basicConfig(level=logging.getLevelName(config.getx("log_level", "INFO")))
    pid = os.getpid()
    config['pid'] = pid
    if "grid_spec" in config:
        logging.basicConfig(level=logging.WARNING)
    logging.info(f"pid is {pid}")
    grid_spec = ""
    if "grid_spec" in config:
        total = config.get_or_default("grid_spec/total", -1)
        current = config.get_or_default("grid_spec/current", -1)
        grid_spec = f"{current:02}/{total:02}/{config['cuda']}#"

    if 'writer_path' not in config or config['writer_path'] is None:
        folder = config['log_tag']
        folder += '-%s' % (config["timestamp_mask"])
        if config["git/state"] == "Good":
            folder += '-%s' % (config['git']['hexsha'][:5])

        config['sub_folder'] = folder
        config['writer_path'] = os.path.join(config['log_folder'],
                                             folder,
                                             config.postfix()
                                             )
    if not os.path.exists(config['writer_path']):
        os.makedirs(config['writer_path'])

    setproctitle(grid_spec + config['writer_path'])

    if 'logfile' not in config or config['logfile']:
        logfile_std = os.path.join(config['writer_path'], "std.log")
        logfile_err = os.path.join(config['writer_path'], "err.log")
        with StdoutTee(logfile_std, buff=1), StderrTee(logfile_err, buff=1):
            try:
                main_run(config)
            except Exception as e:
                with open(os.path.join(config['writer_path'], "runtime-error.txt"), "w") as fp:
                    fp.write(str(e) + "\n")
                sys.stderr.write(str(e) + "\n")
                logging.error(traceback.print_exc())
                raise e
    else:
        main_run(config)
    torch.cuda.empty_cache()
    return


@timer
def main_run(config):
    device = config.getx("device", "cuda")
    logging.info(config)
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    if config.get_or_default('cuda', 'auto') == 'auto':
        import GPUtil
        gpus = GPUtil.getAvailable(order='memory', limit=1)
        assert len(gpus) != 0
        os.environ['CUDA_VISIBLE_DEVICES'] = f"{gpus[0]}"
        logging.info(f"automatically switch to cuda: {gpus[0]}")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = config['cuda']

    torch.manual_seed(config['seed'])
    random.seed(config['seed'])

    with Timer("init cuda"):
        torch.zeros(5, 3).cuda()

    summary = SummaryWriter(config['writer_path'])
    summary.add_text('config', config.__str__())
    logging.info(f"output to {config['writer_path']}")

    kg_dataset = kgdataset_resolver.make(
        config.getx("dataset/kg", None),
        config=config,
        split_mode='all_for_train')
    source_dataset = dataset_resolver.make(
        config.getx("dataset/source", None),
        config=config,
        kg_e_map=kg_dataset.e_map
    )
    target_dataset = dataset_resolver.make(
        config.getx("dataset/target", None),
        config=config,
        split_mode=config.getx("dataset/target_split_mode", 'leave_one_out'),
        kg_e_map=kg_dataset.e_map,
        hist_dataset=source_dataset,
        # k=config.getx("model/k", 8),
    )

    if config.getx("dataset/third", None) is None:
        train_dataset = source_dataset
    else:
        train_dataset = dataset_resolver.make(
            config.getx("dataset/third", None),
            config=config,
            kg_e_map=kg_dataset.e_map,
            # k=config.getx("model/k", 8),
        )

    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=None,
    )

    # 模型定义
    model = model_resolver.make(
        config.getx("model/model", None),
        config=config,
        kg_dataset=kg_dataset
    )

    if "grid_spec" not in config:
        with Timer("loading model and assign GPU memory..."):
            model = model.to(device=device)
    else:
        model = model.to(device=device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())

    # source_evaluator = Evaluator(config, summary, source_dataset, prefix="S_")
    # target_evaluator = Evaluator(config, summary, target_dataset, prefix="T_")
    target_evaluator = FullEvaluator(config, summary, target_dataset, prefix="T_")

    # 优化器
    if config.getx("model/optimizer_class", "Adagrad") == "Adagrad":
        optimizer = Adagrad(model_parameters, **config['model/optimizer'])
    else:
        optimizer_class = op_resolver.lookup(config.getx("model/optimizer_class", "Adagrad"))
        optimizer = optimizer_class(model_parameters, **config['model/optimizer'])

    epoch_loop = range(config['epochs'])
    if config.get_or_default("train/epoch_tqdm", False):
        epoch_loop = tqdm(epoch_loop,
                          desc="train",
                          bar_format="{desc} {percentage:3.0f}%|{bar:10}|{n_fmt:>5}/{total_fmt} "
                                     "[{elapsed}<{remaining} {rate_fmt}{postfix}]",
                          )
    epoch_run = 0
    for epoch in epoch_loop:
        epoch_run = epoch
        # 数据记录和精度验证
        if epoch % config['evaluator_time'] == 0:
            if config.getx("evaluator_args/checkpoint_save", False):
                torch.save({
                    "model": model.state_dict(),
                    "config": config,
                }, os.path.join(config['writer_path'], f"checkpoint-{epoch:04}.tar"))
            # source_evaluator.evaluate(model, epoch)
            target_evaluator.evaluate(model, epoch)

            # if source_evaluator.should_stop() and target_evaluator.should_stop():
            if target_evaluator.should_stop():
                logger.info("early stop...")
                break
        # 我们 propose 的模型训练
        epoch_loss = []
        loader = dataloader
        if config.get_or_default("train/batch_tqdm", True):
            loader = tqdm(loader,
                          desc=f'train  \tepoch: {epoch:05}/{config["epochs"]}',
                          bar_format="{desc}{percentage:3.0f}%|{bar:10}{r_bar}",
                          )
        for batch, pack in enumerate(loader):
            optimizer.zero_grad()
            model.train()

            loss = model(*pack, dataset=source_dataset)
            regular = model.additional_regularization()
            total = loss.sum() + regular

            total.backward()
            optimizer.step()
            epoch_loss.append(loss.mean())

            # assert model.user_embedding.weight.isnan().sum() == 0
            # assert model.entity_embedding.weight.isnan().sum() == 0

            model.batch_hook(epoch=epoch, batch=batch)

        epoch_loss = torch.tensor(epoch_loss).mean()
        summary.add_scalar('Epoch/Loss', epoch_loss, global_step=epoch)
        if isinstance(epoch_loop, tqdm):
            if "grid_spec" in config:
                total = config.get_or_default("grid_spec/total", -1)
                current = config.get_or_default("grid_spec/current", -1)
                epoch_loop.desc = f"{current:04}/{total:04}/{config['cuda']}#"
            else:
                epoch_loop.desc = f"train {epoch_loss:5.4f}"

        model.epoch_hook(epoch=epoch)

    # source_evaluator.record_best(epoch_run)
    target_evaluator.record_best(epoch_run)

    summary.close()
    if config.getx("evaluator_args/best_save", False):
        torch.save({
            "model": model.state_dict(),
            "config": config,
        }, os.path.join(config['writer_path'], f"checkpoint-best.tar"))
