from dataclasses import dataclass
from conf.architecture.architecture import Architecture
from conf.folders.folders import Folders
from conf.estimation.estimation import Estimation
from conf.dataset.dataset import Data


@dataclass(frozen=True)
class Project():
    architecture: Architecture
    folders: Folders
    estimation: Estimation
    data: Data
