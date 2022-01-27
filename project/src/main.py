from tqdm import tqdm
from os import path
import hydra
import torch
from hydra.core.config_store import ConfigStore
from conf.config import Project

from core.train import (crete_result_path,
                        define_model_architecture, 
                        train_model)

from core.data_parser.datahandler import create_loader, create_dataset


cs = ConfigStore.instance()
cs.store(name="conf", node=Project)

@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg: Project) -> None:

    architecture = cfg.architecture
    dataset = cfg.dataset
    estimation = cfg.estimation
    folders = cfg.folders
    problem = cfg.problem

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Executing experiment")
    print(f"Number of epochs: {architecture.epochs}")
    print(f"Learning rate: {architecture.learning_rate}")
    print(f"Architecture: {architecture.hidden_layer_sizes}")
    print(f"Dataset: {dataset.file}")

    filename = "discrete_results_{}_{}_{}.csv".format(architecture.epochs,
                                                      architecture.learning_rate,
                                                      architecture.hidden_layer_sizes)

    result_path = crete_result_path(folders.results.discrete, filename)

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


if __name__ == "__main__":
    main()
