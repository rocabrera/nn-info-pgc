import numpy as np
import pandas as pd
import torch
from models.model import BinaryMLP
from distributions.discrete_stats import get_discrete_mutual_information
from distributions.continuos_stats import get_continuos_mutual_information

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

