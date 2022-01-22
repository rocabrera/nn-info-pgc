import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from networks.custom_dataset import SampleDataset

from networks.models.model import BinaryMLP
from networks.models.architecture import ModelArchitecture

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from distributions.discrete_stats import get_discrete_mutual_information


# bunch = load_breast_cancer(as_frame=True)
# X = bunch["data"]
# y = bunch["target"]

# Prepare Dataset
df = pd.read_csv("data/sample_dataset_circles.csv")
y = df["label"].astype("category").cat.codes
X = df.drop(columns=["label"])


batch_size = len(X)
dataset = SampleDataset(X.to_numpy(), y.to_numpy())
loader = DataLoader(dataset = dataset,
                    shuffle = True,
                    batch_size = batch_size)

# Prepare Model
n_features = len(X.columns)
hidden_layer = [10, 7, 5, 3]
output_size = y.nunique()

# print(f"Tamanho dataset {len(df)}")
# print(f"Número de features {n_features}")
# print(f"Número de neurônios em cada camada escondida {hidden_layer}")
# print(f"Número de classes {output_size}")

activation = nn.ReLU()
architecture = ModelArchitecture(n_features, hidden_layer, output_size, activation)

learning_rate = 0.01

device = "cuda"
model = BinaryMLP(architecture)
model.to(device)

# train the model
def train_model(train_dl, model, rand_init_number, n_bin):
    # define the optimization
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # losses = []
    # accur = []
    for epoch in tqdm(range(1000), desc="Epochs", position=1, leave=False):
        for i, (inputs, targets) in enumerate(train_dl):
            inputs_in_device = inputs.to(device)
            targets_in_device = targets.to(device, dtype=torch.float32)
            
            optimizer.zero_grad() # clear the gradients
            yhat = model(inputs_in_device) # compute the model output
            loss = criterion(yhat, targets_in_device) # calculate loss
            loss.backward() 
            optimizer.step() # update model weights

            with torch.no_grad():
                output_layers = model.get_layers_output()

                for idx_layer, output_layer in enumerate(output_layers):

                    if len(output_layer.shape) == 1:
                        xs = np.hstack([output_layer.reshape(-1,1), inputs.numpy()])
                        ys = np.hstack([output_layer.reshape(-1,1), targets.numpy().reshape(-1,1)])

                        ixt = get_discrete_mutual_information(pd.DataFrame(xs), 
                                                            n_bin=n_bin, 
                                                            n_fst_vector=1)

                        iyt = get_discrete_mutual_information(pd.DataFrame(ys), 
                                                            n_bin=n_bin, 
                                                            n_fst_vector=1)
                    else:
                        xs = np.hstack([output_layer, inputs.numpy()])
                        ys = np.hstack([output_layer, targets.numpy().reshape(-1,1)])
                        n_xs_vector = output_layer.shape[1]
                        ixt = get_discrete_mutual_information(pd.DataFrame(xs), 
                                                            n_bin=n_bin, 
                                                            n_fst_vector=n_xs_vector)

                        iyt = get_discrete_mutual_information(pd.DataFrame(ys), 
                                                            n_bin=n_bin, 
                                                            n_fst_vector=n_xs_vector)
                                                          
                    with open(result_path, "a") as f:
                        f.write(f"{epoch},{rand_init_number},{idx_layer+1},{ixt},{iyt}\n")

            # if epoch%100 == 0:
            #     aux = yhat.cpu().detach().numpy().round()
            #     acc = (aux == targets.numpy()).mean()
            #     losses.append(loss)
            #     accur.append(acc)
            #     print("epoch {}\tloss : {}\t accuracy : {}".format(epoch, loss, acc))

result_path = "resultados_test.csv"

if not os.path.isfile(result_path):
    with open(result_path, "a") as f:
        f.write("epoch,rand_init,layer,T,Y\n")

for idx in tqdm(range(10), desc="Initialization", position=0):
    model = BinaryMLP(architecture)
    model.to(device)
    train_model(loader, model, idx, 5)
