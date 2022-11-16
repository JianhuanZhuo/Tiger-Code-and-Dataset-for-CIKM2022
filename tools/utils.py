import io
import json
import logging
import os
import pickle
import time
import zipfile
from collections import defaultdict
from functools import wraps

import requests
import torch
from tqdm import tqdm

logger = logging.getLogger()


def cache(generator):
    @wraps(generator)
    def decorated(self, *args, **kwargs):
        func_name = generator.__name__
        assert hasattr(self, 'cache_identify_str')
        ids = getattr(self, 'cache_identify_str')
        cache_name = f"cache.{ids}-{func_name}.pkl"
        assert hasattr(self, 'cache_folder')
        cache_folder = getattr(self, 'cache_folder')
        assert hasattr(self, 'cache_abort_if_not_exist')
        cache_abort_if_not_exist = getattr(self, 'cache_abort_if_not_exist')

        return cache_or(
            cache_name=cache_name,
            folder=cache_folder,
            generator=lambda: generator(self, *args, **kwargs),
            abort_if_not_exist=cache_abort_if_not_exist,
        )

    return decorated


GlobalMemoryCaches = dict()


def cache_or(cache_name, folder=None, *, generator: callable, abort_if_not_exist=False, use_json=False):
    extend = ".json" if use_json else ".pkl"
    tool = json if use_json else pickle
    b_flag = '' if use_json else 'b'

    if not cache_name.endswith(extend):
        raise NotImplemented
    if not cache_name.startswith("cache."):
        raise NotImplemented

    abs_name = os.path.abspath(os.path.join(folder, cache_name)) if folder else os.path.abspath(cache_name)

    if abs_name in GlobalMemoryCaches:
        return GlobalMemoryCaches[abs_name]

    zip_file = abs_name + ".zip"
    if os.path.exists(zip_file):
        with Timer(f"extract zipfile: {zip_file}"):
            with zipfile.ZipFile(zip_file, mode='r') as zf:
                assert len(zf.namelist()) == 1 and cache_name in zf.namelist()
                with zf.open(zf.namelist()[0], mode='r') as f:
                    result = tool.load(f)
                    GlobalMemoryCaches[abs_name] = result
                    return result

    if os.path.exists(abs_name):
        with Timer(f"read {cache_name}" if not abort_if_not_exist else None):
            with open(abs_name, 'r' + b_flag) as f:
                result = tool.load(f)
                GlobalMemoryCaches[abs_name] = result
                return result
    elif abort_if_not_exist:
        raise FileNotFoundError(f"cache not found : {abs_name}")
    else:
        with Timer(f"generating data: {cache_name}"):
            result = generator()
        fold = os.path.dirname(abs_name)
        if not os.path.exists(fold):
            os.makedirs(fold, exist_ok=True)
        if not os.path.exists(abs_name):
            with Timer(f"  dump data: {cache_name}"):
                with open(abs_name, 'w' + b_flag) as f:
                    tool.dump(result, f, protocol=4)
        GlobalMemoryCaches[abs_name] = result
        return result


def group_kv(kvs, tqdm_title=None, return_type='set'):
    if return_type == 'set':
        result = defaultdict(set)
        if tqdm_title:
            kvs = tqdm(kvs, desc=tqdm_title)
        for k, v in kvs:
            result[k].add(v)
        return dict(result)
    elif return_type == 'list':
        result = defaultdict(list)
        if tqdm_title:
            kvs = tqdm(kvs, desc=tqdm_title)
        for k, v in kvs:
            result[k].append(v)
        return dict(result)
    else:
        raise NotImplementedError(f"return_type error {return_type}")


def clip_by_norm(value, clip):
    """
    参考：https://www.tensorflow.org/api_docs/python/tf/clip_by_norm
    """
    assert len(value.shape) == 2
    assert type(clip) == float or type(clip) == int
    num, dim = value.shape
    norms = value.norm(p=2, dim=1, keepdim=True)
    assert norms.shape == torch.Size([num, 1])
    clip_mask = (norms > clip).float()
    res = (1 - clip_mask) * value + clip_mask * (value * clip / norms)
    assert res.shape == torch.Size([num, dim])
    return res


def download_zip(file_url, destination):
    print(f"download from {file_url}")
    r = requests.get(file_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    print(f"extractall...")
    z.extractall(destination)
    print(f"download zip file OK")
    # destination = os.path.dirname(file_name)
    # file = os.path.basename(file_name)
    # r = requests.get(file_url, stream=True)
    # total_length = int(r.headers.get('content-length'))
    # z = zipfile.ZipFile(io.BytesIO(r.content))
    # z.extractall(destination)
    #
    # with open(file_name, 'wb') as fp:
    #     for chunk in tqdm(r.iter_content(chunk_size=1024), total=(total_length / 1024) + 1, desc='download'):
    #         if chunk:
    #             fp.write(chunk)
    #             fp.flush()
    # if file.endswith('.zip'):
    #     print(f"extract file: {file_name}")
    #     zipfile.ZipFile(open(file_name, 'rb')).extractall(destination)


def exist_or_download(file_pth, download_url):
    if not os.path.exists(file_pth):
        if os.path.exists(file_pth + ".zip"):
            import zipfile
            with zipfile.ZipFile(file_pth + ".zip", 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(file_pth))
        else:
            download_zip(download_url, os.path.dirname(file_pth))


def timer(func):
    def fun(*args, **kwargs):
        t_start = time.perf_counter()
        result = func(*args, **kwargs)
        t = time.perf_counter() - t_start
        t_str = f"{t:.2f}s" if int(t // 60) == 0 else f"{int(t // 60)}:{t % 60:.2f}s"
        logger.info(f'{func.__name__}:{t_str}')
        return result

    return fun


class Timer(object):
    def __init__(self, desc=''):
        self.t = 0
        self.desc = desc

    def __enter__(self):
        self.t = time.perf_counter()
        if self.desc is not None:
            logger.info(f"{self.desc}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.desc is not None:
            t = time.perf_counter() - self.t
            t_str = f"{t:.2f}s" if int(t // 60) == 0 else f"{int(t // 60)}:{t % 60:.2f}s"
            logger.info(f'  {self.desc}:{t_str}')


if __name__ == '__main__':
    def test1():
        print('func start')
        with Timer("test 1"):
            time.sleep(2)
            print('func end')


    @timer
    def test2():
        print('func start')
        time.sleep(2)
        print('func end')


    test1()
    test2()
