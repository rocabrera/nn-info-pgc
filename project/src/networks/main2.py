import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from project.src.custom_dataset import BeansDataset

from models.model import MLP
from models.architecture import ModelArchitecture

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# Prepare Dataset
df = pd.read_excel(os.path.join("data", "Dry_Bean_Dataset.xlsx"),skipfooter=3611)
y = df["Class"].astype("category").cat.codes
X = df.drop(columns=["Class"])

batch_size = 1000
dataset = BeansDataset(X.to_numpy(), y.to_numpy())
loader = DataLoader(dataset = dataset,
                    shuffle = True,
                    batch_size = batch_size)


# Prepare Model
n_features = len(X.columns)
hidden_layer = [10]
output_size = y.nunique()

print(f"Tamanho dataset {len(df)}")

print(f"Número de features {n_features}")
print(f"Número de neurônios em cada camada escondida {hidden_layer}")
print(f"Número de classes {output_size}")

activation = nn.ReLU()
architecture =  ModelArchitecture(n_features, hidden_layer, output_size, activation)

learning_rate = 0.01

device = "cuda"
model = MLP(architecture)
model.to(device)

# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # enumerate epochs
    for epoch in tqdm(range(100), desc="Epochs", position=1, leave=False):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            inputs_in_device = inputs.to(device)
            targets_in_device = targets.to(device, dtype=torch.int64)
            
            optimizer.zero_grad() # clear the gradients
            yhat = model(inputs_in_device) # compute the model output
            print(yhat.shape)
            loss = criterion(yhat, targets_in_device) # calculate loss
            loss.backward() # credit assignment
            optimizer.step() # update model weights

            with torch.no_grad():
                output_layers = model.get_layers_output()

                # I(T, y)

                output_layers = [output_layer.shape for output_layer in output_layers]
        
    return output_layers

#for _ in tqdm(range(3), desc="Initialization", position=0):
output_layers = train_model(loader, model)
print(output_layers)
#    break