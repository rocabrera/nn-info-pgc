from tqdm import tqdm
from models.model import BinaryMLP
from models.architecture import ModelArchitecture

import torch.nn as nn
from core.data_parser.datahandler import create_loaders, create_dataset
from core.trainer import run_experiment


def define_model_architecture(n_features, hidden_layer, n_output, device):

    activation = nn.ReLU()
    architecture = ModelArchitecture(n_features, hidden_layer, n_output, activation)
    model = BinaryMLP(architecture)
    model.to(device)

    return model


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

            model = define_model_architecture(n_features, 
                                              architecture.hidden_layer_sizes, 
                                              n_output, 
                                              device)
                                        
            run_experiment(train_dl = train_dataloader, 
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