import torch as tf
import pandas as pd
from os.path import splitext, join
from core.data_parser.datasets import DefaultDataset
from torch.utils.data import DataLoader


def create_dataset(data_folder, dataset_file_name, sample_size_pct):

    _, ext = splitext(dataset_file_name)
    df = pd.read_csv(join(data_folder, dataset_file_name))
    n_samples = len(df)
    sample_size = int(sample_size_pct*n_samples)
    if sample_size <= n_samples:
        df = df.sample(sample_size) 
    if dataset_file_name == "breast_cancer.csv":
        
        y = df["target"].astype("category").cat.codes
        X = df.drop(columns=["target"])
        dataset = DefaultDataset(X.to_numpy(), y.to_numpy())
        
    elif ext == ".csv":
        y = df["label"].astype("category").cat.codes
        X = df.drop(columns=["label"])

        dataset = DefaultDataset(X.to_numpy(), y.to_numpy())
    
    return len(y.unique()), X.shape[1], dataset


def create_loaders(dataset, batch_pct:float=1.0, valid_pct=0.1):


    dataset_size = len(dataset)
    valid_n = int(valid_pct*dataset_size)

    valid_dataset, train_dataset = tf.utils.data.random_split(dataset, [valid_n, (dataset_size-valid_n)])

    train_batch_size = int(len(train_dataset)*batch_pct) 
    train_dataloader = DataLoader(dataset = train_dataset,
                                  shuffle = True,
                                  batch_size = train_batch_size)

    valid_batch_size = int(len(valid_dataset)*batch_pct) 
    valid_dataloader = DataLoader(dataset = valid_dataset,
                                  shuffle = False,
                                  batch_size = valid_batch_size)

    return train_dataloader, valid_dataloader 