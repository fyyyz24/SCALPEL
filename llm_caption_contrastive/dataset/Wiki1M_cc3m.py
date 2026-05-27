from llm2vec.dataset.dataset import DataSample, TrainSample, Dataset
import numpy as np
from accelerate.logging import get_logger
import pandas as pd
import datasets
logger = get_logger(__name__, log_level="INFO")


class Captions(Dataset):
    def __init__(
        self,
        dataset_name: str = "captions",
        split: str = "validation",
        file_path: str = 'cc3m.csv',
        wiki1m=None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.data = []
        if wiki1m is not None:
            self.data = wiki1m.data
        self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: str = None):
        logger.info(f"Loading captions data from {file_path}...")
        id_ = len(self.data)
        cc3m = pd.read_csv(file_path)
        data = cc3m[['short_caption', 'long_caption']].values

        # 使用 NumPy 随机选取每一行中的一个字符串值
        selected_values = np.choose(np.random.randint(2, size=len(data)), data.T)
        for line in selected_values:
            line = line.strip()
            self.data.append(
                DataSample(
                    id_=id_,
                    query=line,
                    positive=line,
                )
            )
            id_ += 1
        logger.info(f"Loaded {len(self.data)} samples.")

    def __getitem__(self, index):
        sample = self.data[index]
        if self.split == "train":
            return TrainSample(texts=[sample.query, sample.positive], label=1.0)
        elif self.split == "validation":
            assert False, "Wiki1M does not have a validation split."
def get_cc3m_captions(file_path: str = 'cc3m.csv'):
    cc3m = pd.read_csv(file_path)
    data = cc3m[['short_caption', 'long_caption']].values
    selected_values = np.choose(np.random.randint(2, size=len(data)), data.T)
    df = pd.DataFrame(selected_values, columns=['text'])
    cc3m = datasets.Dataset.from_pandas(df)
    return cc3m
def merge_cc3m_wikiraw103(cc3m=None, wiki1m:datasets.Dataset=None):
    train_ratio = wiki1m['train'].num_rows / sum([wiki1m['train'].num_rows, wiki1m['validation'].num_rows, wiki1m['test'].num_rows])
    cc3m = cc3m.train_test_split(train_size=train_ratio,seed=42)
    new_train = datasets.concatenate_datasets([cc3m['train'], wiki1m['train']])
    new_val = datasets.concatenate_datasets([cc3m['test'], wiki1m['validation']])
    new_dataset = datasets.DatasetDict({'train':new_train, 'validation':new_val})
    logger.info(f"New dataset created with {new_dataset['train'].num_rows} training samples and {new_dataset['validation'].num_rows} validation samples.")
    return new_dataset