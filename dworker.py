import queue
import socket
import sys
from multiprocessing import Pool
from multiprocessing.managers import BaseManager

from full_trainer import wrap as full_trainer

server_addr = 'worker-0'
server_port = 4135
server_keys = b'4135'

# WORK_NAME_MAP = {
#     'az-eus-p100-worker-20028':
# }

worker = socket.gethostname()[-12:]


# 创建类似的QueueManager:
class QueueManager(BaseManager):
    pass


def gpu_worker(gpu, trainer):
    # 由于这个QueueManager只从网络上获取Queue，所以注册时只提供名字:
    QueueManager.register('get_task_queue')
    QueueManager.register('get_feed_queue')
    # 连接到服务器，也就是运行task_master.py的机器:
    print('Connect to server %s...' % server_addr)
    # 端口和验证码注意保持与task_master.py设置的完全一致:
    m = QueueManager(address=(server_addr, server_port), authkey=server_keys)
    # 从网络连接:
    m.connect()
    # 获取Queue的对象:
    task = m.get_task_queue()
    feed = m.get_feed_queue()
    # 从task队列取任务,并把结果写入result队列:

    while not task.empty():
        try:
            cfg = task.get()
            cfg['cuda'] = gpu
            trainer(cfg)
            feed.put(f"{worker}:{gpu}")
        except queue.Empty:
            print('task queue is empty.')
    # 处理结束:
    print('worker exit.')


def multi_worker(gpus, trainer=full_trainer):
    process_pool = Pool(len(gpus))
    for gpu in gpus:
        process_pool.apply_async(gpu_worker, args=(str(gpu), trainer))
    process_pool.close()
    process_pool.join()


if __name__ == '__main__':
    t = 1
    if len(sys.argv) > 1:
        t = int(sys.argv[1])
    import GPUtil
    # gpus = GPUtil.getAvailable(order='memory', limit=1)
    gpu_list = list(range(len(GPUtil.getGPUs())))
    # gpu_list = [0, 1, 2, 3]
    multi_worker(gpu_list * t)
