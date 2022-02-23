import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from models.model import BinaryMLP
from models.architecture import ModelArchitecture

import torch
import torch.nn as nn
from torchmetrics.functional import auroc

from core.data_parser.datahandler import create_loader, create_dataset
from distributions.discrete_stats import get_discrete_mutual_information
from distributions.continuos_stats import get_continuos_mutual_information


def define_model_architecture(n_features, hidden_layer, n_output, device):

    activation = nn.ReLU()
    architecture = ModelArchitecture(n_features, hidden_layer, n_output, activation)
    model = BinaryMLP(architecture)
    model.to(device)

    return model

# train the model
def train_model(train_dl, 
                model, 
                n_epochs, 
                learning_rate, 
                rand_init_number, 
                estimation_param, 
                result_file_path, 
                device,
                discrete):
    # define the optimization
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    for epoch in tqdm(range(n_epochs), desc="Epochs", position=1, leave=False):

        for i, (inputs, targets) in enumerate(train_dl):
            inputs_in_device = inputs.to(device)
            targets_in_device = targets.to(device, dtype=torch.float32)
            
            optimizer.zero_grad() # clear the gradients
            yhat = model(inputs_in_device) # compute the model output
            loss = criterion(yhat, targets_in_device) # calculate loss
            loss.backward() 
            optimizer.step()

        aucroc_result = auroc(yhat, targets_in_device.to(torch.int), pos_label=1)
        if discrete:
            make_discrete_information_plane(model=model, 
                                            inputs=inputs_in_device, 
                                            targets=targets_in_device, 
                                            auc=aucroc_result.cpu().numpy(),
                                            n_bin=estimation_param, 
                                            result_file_path=result_file_path, 
                                            epoch=epoch, 
                                            rand_init_number=rand_init_number,)
        else:
            make_continuos_information_plane(model=model, 
                                             inputs=inputs_in_device, 
                                             targets=targets_in_device, 
                                             auc=aucroc_result.cpu().numpy(),
                                             kernel_size=estimation_param, 
                                             result_file_path=result_file_path, 
                                             epoch=epoch, 
                                             rand_init_number=rand_init_number)
            

def make_continuos_information_plane(model: BinaryMLP, 
                                     inputs: torch.Tensor, 
                                     targets: torch.Tensor, 
                                     auc: float, 
                                     kernel_size: int, 
                                     result_file_path: str, 
                                     epoch: int, 
                                     rand_init_number: int):

    with torch.no_grad():
        output_layers = model.get_layers_output()

        for idx_layer, output_layer in enumerate(output_layers):

            if len(output_layer.shape) == 1:
                xs = np.hstack([output_layer.reshape(-1,1), inputs.cpu().numpy()])
                ys = np.hstack([output_layer.reshape(-1,1), targets.cpu().numpy().reshape(-1,1)])
                n_xs_vector = 1
            else:
                xs = np.hstack([output_layer, inputs.cpu().numpy()])
                ys = np.hstack([output_layer, targets.cpu().numpy().reshape(-1,1)])
                n_xs_vector = output_layer.shape[1]
            
            ixt = get_continuos_mutual_information(pd.DataFrame(xs), 
                                                   kernel_size=kernel_size, 
                                                   n_fst_vector=n_xs_vector)

            iyt = get_continuos_mutual_information(pd.DataFrame(ys), 
                                                   kernel_size=kernel_size, 
                                                   n_fst_vector=n_xs_vector)
                                                    
            with open(result_file_path, "a") as f:
                f.write(f"{epoch};{rand_init_number};{idx_layer+1};{ixt};{iyt};{auc}\n")


def make_discrete_information_plane(model: BinaryMLP, 
                                    inputs: torch.Tensor, 
                                    targets: torch.Tensor, 
                                    auc: float, 
                                    n_bin: int, 
                                    result_file_path: str, 
                                    epoch: int, 
                                    rand_init_number: int) -> None:

    with torch.no_grad():
        output_layers = model.get_layers_output()

        for idx_layer, output_layer in enumerate(output_layers):

            if len(output_layer.shape) == 1:
                xs = np.hstack([output_layer.reshape(-1,1), inputs.cpu().numpy()])
                ys = np.hstack([output_layer.reshape(-1,1), targets.cpu().numpy().reshape(-1,1)])
                n_xs_vector = 1
            else:
                xs = np.hstack([output_layer, inputs.cpu().numpy()])
                ys = np.hstack([output_layer, targets.cpu().numpy().reshape(-1,1)])
                n_xs_vector = output_layer.shape[1]

            ixt = get_discrete_mutual_information(pd.DataFrame(xs), 
                                                  n_bin=n_bin, 
                                                  n_fst_vector=n_xs_vector)

            iyt = get_discrete_mutual_information(pd.DataFrame(ys), 
                                                  n_bin=n_bin, 
                                                  n_fst_vector=n_xs_vector)
                                                    
            with open(result_file_path, "a") as f:
                f.write(f"{epoch};{rand_init_number};{idx_layer+1};{ixt};{iyt};{auc}\n")


def execute_experiment(architecture, 
                       dataset, 
                       problem, 
                       estimation_param,
                       sample_size_pct, 
                       device:str, 
                       folders_data:str,
                       result_file_path:str,
                       discrete:bool) -> None:

    n_output, n_features, torch_dataset = create_dataset(folders_data, 
                                                         dataset.file,
                                                         sample_size_pct)

    dataloader = create_loader(torch_dataset)

    try:
        for idx in tqdm(range(1, problem.n_init_random + 1), desc="Initialization", position=0):

            model = define_model_architecture(n_features, 
                                              architecture.hidden_layer_sizes, 
                                              n_output, 
                                              device)
                                        
            train_model(train_dl = dataloader, 
                        model = model, 
                        n_epochs = architecture.epochs,
                        learning_rate = architecture.learning_rate,
                        rand_init_number = idx,
                        estimation_param = estimation_param,
                        result_file_path = result_file_path,
                        device = device,
                        discrete=discrete)

    except Exception as e:
        raise e("An unmapped error occurred while running the experiment.")