from class_resolver import Resolver
from torch.utils.data import Dataset
from dataset.ML100K_dataset import ML100KDataset
from dataset.ML1M_dataset import ML1MDataset
from dataset.ML10M_dataset import ML10MDataset
from dataset.LastFM_dataset import LastFMDataset
from dataset.AmazonInstantVideo_dataset import AIVDataset
from dataset.Yelp2018_dataset import Yelp2018Dataset
from dataset.bmm_kg_dataset import BMMKGDataset
from dataset.AB202Kdc3ge_dataset import AB202Kdc3geDataset, AB202Kdc3geAvgDataset, AB202Kdc3Dataset
from dataset.AM143Kdc3ge_dataset import AM143Kdc3Dataset, AM143Kdc3geDataset, AM143Kdc3geAvgDataset
from dataset.ML1Mdc5ge_dataset import ML1Mdc5geDataset, ML1Mdc5geAvgDataset
from dataset.lfm1Mdc5ge_dataset import LFM1Mdc5geAvgDataset, LFM1Mdc5geDataset
from dataset.AB202Kdc3geML1Mdc5ge_dataset import AB202Kdc3geML1Mdc5geDataset, AB202Kdc3geML1Mdc5geAvgDataset
from dataset.AM143Kdc3geML1Mdc5ge_dataset import AM143Kdc3geML1Mdc5geDataset, AM143Kdc3geML1Mdc5geAvgDataset
from dataset.AB202Kdc3geLFM1Mdc5ge_dataset import AB202Kdc3geLFM1Mdc5geDataset, AB202Kdc3geLFM1Mdc5geAvgDataset
from dataset.AM143Kdc3geLFM1Mdc5ge_dataset import AM143Kdc3geLFM1Mdc5geDataset, AM143Kdc3geLFM1Mdc5geAvgDataset

from dataset.kg_dataset import KGDataset, BSKGDataset
from dataset.BBM_hop1_connect_kg_dataset import BBMhop1ConnectKGDataset
from dataset.MBM_hop1_connect_kg_dataset import MBMhop1ConnectKGDataset
from dataset.LBM_hop1_connect_kg_dataset import LBMhop1ConnectKGDataset
from dataset.LMBM_hop1_connect_kg_dataset import LMBMhop1ConnectKGDataset

dataset_resolver = Resolver(
    {
        ML100KDataset,
        ML1MDataset,
        ML10MDataset,
        LastFMDataset,
        AIVDataset,
        Yelp2018Dataset,
        AB202Kdc3Dataset,
        AB202Kdc3geDataset,
        AM143Kdc3Dataset,
        AM143Kdc3geDataset,
        AM143Kdc3geAvgDataset,
        AB202Kdc3geAvgDataset,
        ML1Mdc5geDataset,
        ML1Mdc5geAvgDataset,
        LFM1Mdc5geDataset,
        LFM1Mdc5geAvgDataset,
        AB202Kdc3geML1Mdc5geDataset,
        AB202Kdc3geML1Mdc5geAvgDataset,
        AM143Kdc3geML1Mdc5geDataset,
        AM143Kdc3geML1Mdc5geAvgDataset,
        AB202Kdc3geLFM1Mdc5geDataset,
        AB202Kdc3geLFM1Mdc5geAvgDataset,
        AM143Kdc3geLFM1Mdc5geDataset,
        AM143Kdc3geLFM1Mdc5geAvgDataset,
    },
    base=Dataset,  # type: ignore
    default=ML100KDataset,
    suffix='dataset',
)

kgdataset_resolver = Resolver(
    {
        KGDataset,
        BSKGDataset,
        BMMKGDataset,
        BBMhop1ConnectKGDataset,
        MBMhop1ConnectKGDataset,
        LBMhop1ConnectKGDataset,
        LMBMhop1ConnectKGDataset,
    },
    base=Dataset,  # type: ignore
    default=BMMKGDataset,
    suffix='dataset',
)
