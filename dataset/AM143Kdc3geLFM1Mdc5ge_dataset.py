import os.path

from dataset import ML100KDataset
from dataset.cf_hist_avg_adaptor import CFHistAdaptor
from dataset.cf_kg_adaptor import CFKGAdaptor
from tools.config import load_config

# 这个数据集是从 AmazonBook 数据集中筛选出和 AmazonMovieTV 有相同用户，且 item 都 KG linkable，且 item 为 5-Core 的
# https://1.gcr:8888/notebooks/notebooks/mine/KG-dataset/Amazon-Book-Movie-5Core
# /%E5%8F%AA%E4%BF%9D%E7%95%99%E5%8F%AF%E9%93%BE%E6%8E%A5%20KG%20%E7%9A%84%20Book.ipynb
from tools.utils import download_zip


class AM143Kdc3geLFM1Mdc5geDataset(CFKGAdaptor, ML100KDataset):
    def cfg_name(self):
        return "AM143Kdc3geLFM1Mdc5ge", "AM143Kdc3geLFM1Mdc5ge.csv", "\t"

    def fetch_dataset(self, rating_file):
        url = "https://github.com/JianhuanZhuo/dataset-package/releases/download/v0.0.2/" \
              "AM143Kdc3geLFM1Mdc5ge.csv.zip"
        download_zip(url, os.path.dirname(rating_file))

    @property
    def record_count(self):
        return 1149034


class AM143Kdc3geLFM1Mdc5geAvgDataset(CFHistAdaptor, AM143Kdc3geLFM1Mdc5geDataset):
    pass


if __name__ == '__main__':
    """
    #user : 11275
    #item : 47449
    #train: 180107
    #valid: 11275
    #tests: 11275
    """
    import logging

    logging.basicConfig(level=logging.INFO)
    cfg = load_config("../config.yaml")
    dataset1 = AM143Kdc3geLFM1Mdc5geDataset(config=cfg)
    # dataset2 = ML10MDataset(config=cfg)
    # dataset2 = ML10MDataset(config=cfg)
    # dataset2 = ML10MDataset(config=cfg)
