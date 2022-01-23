import hydra
from hydra.core.config_store import ConfigStore
from conf.config import Project

from src.networks.train import train_model
from src.custom_dataset import Dataset

cs = ConfigStore.instance()
cs.store(name="conf", node=Project)

@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg: Project) -> None:

    architecture = cfg.architecture
    dataset = cfg.dataset
    estimation = cfg.estimation
    folders = cfg.folders


    train_model()

if __name__ == "__main__":
    main()
