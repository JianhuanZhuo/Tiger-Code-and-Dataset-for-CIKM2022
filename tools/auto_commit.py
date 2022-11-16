import socket
import datetime
import logging
import os.path
import random
from git import Repo

logger = logging.getLogger()
# logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)


def auto_record_output(folder):
    repo = Repo("../output")
    repo.index.add(folder)
    repo.index.commit(f"auto-record on {socket.gethostname()}")
    repo.remote("origin").push("master")

