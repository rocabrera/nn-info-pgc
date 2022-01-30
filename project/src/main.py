import hydra
import torch
from hydra.core.config_store import ConfigStore
from conf.config import Project

from core.train import execute_discrete_experiment

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
    print()

    execute_discrete_experiment(architecture=architecture, 
                                folders=folders, 
                                dataset=dataset, 
                                problem=problem, 
                                estimation=estimation, 
                                device=device)


if __name__ == "__main__":
    main()
