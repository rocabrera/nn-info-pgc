import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from os import path

from models.model import BinaryMLP
from models.architecture import ModelArchitecture

import torch
import torch.nn as nn
from core.data_parser.datahandler import create_loader, create_dataset
from distributions.discrete_stats import get_discrete_mutual_information


def write_file():
    pass


def define_model_architecture(n_features, hidden_layer, n_output, device):

    activation = nn.ReLU()
    architecture = ModelArchitecture(n_features, hidden_layer, n_output, activation)
    model = BinaryMLP(architecture)
    model.to(device)

    return model

# train the model
def train_model(train_dl, model, n_epochs, learning_rate, rand_init_number, n_bin, result_path, device):
    # define the optimization
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # losses = []
    # accur = []
    for epoch in tqdm(range(n_epochs), desc="Epochs", position=1, leave=False):
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

def crete_result_path(folder_results:str, filename:str):

    Path(folder_results).mkdir(parents=True, exist_ok=True)

    result_path = path.join(folder_results, filename)
    if not path.isfile(result_path):
        with open(result_path, "a") as f:
            f.write("epoch,rand_init,layer,T,Y\n")
    else:
        raise Exception("Experiment already done.")
    
    return result_path


def execute_discrete_experiment(architecture, folders, dataset, problem, estimation, device):

    string_arch = str(architecture.hidden_layer_sizes).strip('[]').replace(' ', '')
    filename = "bins{}_epochs{}_arch{}_lr{}.csv".format(estimation.discrete.bins,
                                                            architecture.epochs,
                                                            string_arch,
                                                            architecture.learning_rate)

    result_path = crete_result_path(folders.results.discrete,
                                    filename)

    n_output, n_features, dataset = create_dataset(folders.data, 
                                                   dataset.file)
    dataloader = create_loader(dataset)

    model = define_model_architecture(n_features, 
                                      architecture.hidden_layer_sizes, 
                                      n_output, 
                                      device)
          
    for idx in tqdm(range(1, problem.n_init_random + 1), desc="Initialization", position=0):
        train_model(train_dl = dataloader, 
                    model = model, 
                    n_epochs = architecture.epochs,
                    learning_rate = architecture.learning_rate,
                    rand_init_number = idx,
                    n_bin = estimation.discrete.bins,
                    result_path = result_path,
                    device = device)