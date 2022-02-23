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


def create_loader(dataset, batch_pct:float=1.0):

    batch_size = int(len(dataset)*batch_pct) 

    dataloader = DataLoader(dataset = dataset,
                            shuffle = True,
                            batch_size = batch_size)
    return dataloader 