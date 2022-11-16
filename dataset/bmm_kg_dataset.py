import os
import logging
import os

from tqdm.auto import tqdm

from dataset.kg_dataset import BSKGDataset
from tools.config import load_config
from tools.utils import download_zip, Timer

logger = logging.getLogger()


class BMMKGDataset(BSKGDataset):
    @property
    def raw_folder(self):
        return "BMMKG107M"

    @property
    def raw_file(self):
        return "bbm_remapped.kg.csv"

    @property
    def remapped(self):
        return True

    @property
    def record_count(self):
        return 107228399

    def fetch_dataset(self, rating_file):
        url = "https://github.com/JianhuanZhuo/dataset-package/releases/download/v0.0.1/"\
              "bbm_remapped.kg.csv.zip"
        download_zip(url, os.path.dirname(rating_file))

    def load_raw(self):
        rating_file = os.path.join(self.folder, self.raw_file)
        if not os.path.exists(rating_file):
            if os.path.exists(rating_file + ".zip"):
                import zipfile
                with zipfile.ZipFile(rating_file + ".zip", 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(rating_file))
            else:
                self.fetch_dataset(rating_file)

        sps = []
        with tqdm(total=self.record_count, desc='load_raw') as pbar:
            with open(rating_file) as infile:
                while True:
                    lines = infile.readlines(1024**2*10)
                    if not lines:
                        break
                    c = 0
                    for line in lines:
                        h, r, t = line.strip().split(self.sep)
                        sps.append((int(h), int(r), int(t)))
                        c += 1
                    pbar.update(c)
            ############################################################################
            # lines = [s for s in tqdm(fp, 'load_raw', total=107228399)]
            # sps = [
            #     [int(x) for x in line.strip().split(self.sep)]
            #     for line in tqdm(lines, 'load_raw', total=107228399)
            # ]
            ############################################################################
            # for line in tqdm(lines, 'load_raw', total=107228399):
            #     sp = line.strip().split(self.sep)
            #     if len(sp) != 3:
            #         continue
            #     h, r, t = sp
            #     sps.append((int(h), int(r), int(t)))
        assert self.remapped
        e_num = max([e for h, r, t in sps for e in (h, t)]) + 1
        r_num = max([r for h, r, t in sps]) + 1

        train, valid, tests = self.split_random_by_proportion(sps)
        return train, valid, tests, e_num, r_num


if __name__ == '__main__':
    """
    #user : 15897
    #item : 39768
    #train: 194417
    #valid: 15897
    #tests: 15897
    """
    logging.basicConfig(level=logging.INFO)
    cfg = load_config("../config.yaml")
    with Timer(f"setup BMMKGDataset"):
        dataset1 = BMMKGDataset(config=cfg)
    print("over")
