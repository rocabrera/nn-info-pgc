import numpy as np
import pandas as pd
from tqdm import tqdm

from models.model import BinaryMLP
from models.architecture import ModelArchitecture

import torch
import torch.nn as nn
from torchmetrics.functional import auroc

from core.data_parser.datahandler import create_loaders, create_dataset
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
                valid_dl,
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
    
    train_loss_batchs = []
    train_auc_batchs = [] 
    valid_loss_batchs = []
    valid_auc_batchs = [] 

    for epoch in tqdm(range(n_epochs), desc="Epochs", position=1, leave=False):

        for _, (inputs, targets) in enumerate(train_dl):
            inputs_train_in_device = inputs.to(device)
            targets_train_in_device = targets.to(device, dtype=torch.float32)
            
            optimizer.zero_grad() # clear the gradients
            yhat = model(inputs_train_in_device) # compute the model output
            train_loss = criterion(yhat, targets_train_in_device) # calculate loss
            train_loss.backward() 
            optimizer.step()

            train_auc_batchs.append(auroc(yhat, targets_train_in_device.to(torch.int), pos_label=1)
                                    .cpu()
                                    .numpy())

            train_loss_batchs.append(train_loss.item())

        train_auc_epoch = np.mean(train_auc_batchs)
        train_loss_epoch = np.mean(train_loss_batchs)


        model.eval()
        with torch.no_grad():
            for input_valid, target_valid in valid_dl:
                input_valid_in_device = input_valid.to(device)
                targets_valid_in_device = target_valid.to(device, dtype=torch.float32)
                yhat_valid = model(input_valid_in_device)
                valid_loss = criterion(yhat_valid, targets_valid_in_device)
                valid_auc_batchs.append(auroc(yhat_valid, targets_valid_in_device.to(torch.int), pos_label=1)
                                        .cpu()
                                        .numpy())

                valid_loss_batchs.append(valid_loss.item())

        valid_auc_epoch = np.mean(valid_auc_batchs)
        valid_loss_epoch = np.mean(valid_loss_batchs)



        _ = model(inputs_train_in_device) # compute the model output
        
        if discrete:
            make_discrete_information_plane(model=model, 
                                            inputs=inputs_train_in_device, 
                                            targets=targets_train_in_device, 
                                            valid_auc=valid_auc_epoch,
                                            train_auc=train_auc_epoch,
                                            valid_loss=valid_loss_epoch,
                                            train_loss=train_loss_epoch,
                                            n_bin=estimation_param, 
                                            result_file_path=result_file_path, 
                                            epoch=epoch, 
                                            rand_init_number=rand_init_number,)
        else:
            make_continuos_information_plane(model=model, 
                                             inputs=inputs_train_in_device, 
                                             targets=targets_train_in_device, 
                                             valid_auc=valid_auc_epoch,
                                             train_auc=train_auc_epoch,
                                             valid_loss=valid_loss_epoch,
                                             train_loss=train_loss_epoch,
                                             kernel_size=estimation_param, 
                                             result_file_path=result_file_path, 
                                             epoch=epoch, 
                                             rand_init_number=rand_init_number)
            

def make_continuos_information_plane(model: BinaryMLP, 
                                     inputs: torch.Tensor, 
                                     targets: torch.Tensor, 
                                     valid_auc:float,
                                     train_auc:float,
                                     valid_loss:float,
                                     train_loss:float,
                                     kernel_size: int, 
                                     result_file_path: str, 
                                     epoch: int, 
                                     rand_init_number: int):

    with torch.no_grad():
        output_layers = model.get_layers_output()
        #T: [(1000, 12), (1000,7), (1000, 3)]

        for idx_layer, output_layer in enumerate(output_layers):

            if len(output_layer.shape) == 1:
                xs = np.hstack([output_layer.reshape(-1,1), inputs.cpu().numpy()])
                ys = np.hstack([output_layer.reshape(-1,1), targets.cpu().numpy().reshape(-1,1)])
                n_xs_vector = 1
            else:
                xs = np.hstack([output_layer, inputs.cpu().numpy()])
                ys = np.hstack([output_layer, targets.cpu().numpy().reshape(-1,1)])
                n_xs_vector = output_layer.shape[1]
            
            ixt = get_continuos_mutual_information(array=xs, 
                                                   kernel_size=kernel_size, 
                                                   n_fst_vector=n_xs_vector)

            iyt = get_continuos_mutual_information(array=ys, 
                                                   kernel_size=kernel_size, 
                                                   n_fst_vector=n_xs_vector)
                                                    
            with open(result_file_path, "a") as f:
                f.write(f"{epoch};{rand_init_number};{idx_layer+1};{ixt};{iyt};{valid_auc};{train_auc};{valid_loss};{train_loss}\n")



def make_discrete_information_plane(model: BinaryMLP, 
                                    inputs: torch.Tensor, 
                                    targets: torch.Tensor, 
                                    valid_auc:float,
                                    train_auc:float,
                                    valid_loss:float,
                                    train_loss:float,
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
                f.write(f"{epoch};{rand_init_number};{idx_layer+1};{ixt};{iyt};{valid_auc};{train_auc};{valid_loss};{train_loss}\n")


def execute_experiment(architecture, 
                       dataset, 
                       problem, 
                       estimation_param,
                       sample_size_pct, 
                       batch_pct,
                       valid_pct,
                       device:str, 
                       folders_data:str,
                       result_file_path:str,
                       discrete:bool) -> None:

    n_output, n_features, torch_dataset = create_dataset(folders_data, 
                                                         dataset.file,
                                                         sample_size_pct)

    train_dataloader, valid_dataloader = create_loaders(dataset=torch_dataset, 
                                                        batch_pct=batch_pct, 
                                                        valid_pct=valid_pct)

    try:
        for idx in tqdm(range(1, problem.n_init_random + 1), desc="Initialization", position=0):


            """
            model tem a função get_layers_output. 
            Talvez implementar uma função no modelo que me retorna os shapes
            do output dos layers... get_layers_shapes, o qual eu calculo
            """

            model = define_model_architecture(n_features, 
                                              architecture.hidden_layer_sizes, 
                                              n_output, 
                                              device)
                                        
            train_model(train_dl = train_dataloader, 
                        valid_dl = valid_dataloader,
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