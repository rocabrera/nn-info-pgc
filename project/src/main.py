import os
import hydra
import torch
import logging
from time import time
from hydra.core.config_store import ConfigStore
from conf.config import Project

from misc.make_figures import create_figures, create_gif
from core.train import execute_experiment
from core.path_creater import (crete_result_folder, 
                               crete_image_folder, 
                               crete_gif_folder,
                               write_header_result_experiment)


log = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)


cs = ConfigStore.instance()
cs.store(name="conf", node=Project)


@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg: Project) -> None:

    architecture = cfg.architecture
    dataset = cfg.dataset
    estimation = cfg.estimation
    folders = cfg.folders
    figures = cfg.figures
    problem = cfg.problem
    discrete = problem.discrete

    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("Executing experiment")
    log.info(f"Number of epochs: {architecture.epochs}")
    log.info(f"Learning rate: {architecture.learning_rate}")
    log.info(f"Architecture: {architecture.hidden_layer_sizes}")
    log.info(f"Dataset: {dataset.file}")

    (gifs_root_path,
     figures_root_path, 
     results_root_folder, 
     estimation_param, 
     result_filename, 
     dataset_name, 
     sample_size_pct) = setup_parameters(discrete=discrete,
                                         estimation=estimation,
                                         architecture=architecture,
                                         folders=folders,
                                         dataset=dataset)

    process_begin_time = time()                                         
    try:
        experiment_time = time()                                         
        log.info(f"Starting experiment.")
        result_folder_path = crete_result_folder(result_root_folder=results_root_folder,
                                                 dataset_name=dataset_name)
        log.info(f"Result Folder created successfully at: {result_folder_path}")
        result_file_path = write_header_result_experiment(result_folder=result_folder_path,
                                                          result_filename=result_filename)
        execute_experiment(architecture=architecture, 
                           dataset=dataset, 
                           problem=problem, 
                           estimation_param=estimation_param, 
                           sample_size_pct=sample_size_pct,
                           device=device,
                           folders_data=folders.data,
                           result_file_path=result_file_path,
                           discrete=discrete)
        experiment_elapsed_time = time() - experiment_time
        log.info(f"Experiment done. Elapsed time: {round(experiment_elapsed_time,2)}")

        figures_time = time()    
        log.info(f"Creating figures.")
        image_folder_path = crete_image_folder(figure_root_folder=figures_root_path,
                                               dataset_name=dataset_name, 
                                               result_filename=result_filename)
        log.info(f"Image Folder created successfully at: {image_folder_path}")
        create_figures(result_file_path=result_file_path,
                       epochs=architecture.epochs,
                       save_folderpath=image_folder_path,
                       **figures.scatterplot_aesthetics) 
        figures_elapsed_time = time() - figures_time
        log.info(f"Figures done. Elapsed time: {round(figures_elapsed_time,2)}")

        gifs_time = time()    
        log.info(f"Creating gifs.")
        gif_folder_path = crete_gif_folder(gif_root_folder=gifs_root_path,
                                           dataset_name=dataset_name)
        log.info(f"Gif Folder created successfully at: {gif_folder_path}")
        create_gif(image_folder_path=image_folder_path,
                   gif_root_path=gif_folder_path, 
                   result_filename=result_filename)
        gif_elapsed_time = time() - gifs_time
        log.info(f"Gif done. Elapsed time: {round(gif_elapsed_time,2)}")

    except Exception as e:
        logging.error(e, exc_info=True)

    finally:
        total_elapsed_time = time() - process_begin_time
        log.info(f"Total elapsed time: {round(total_elapsed_time)}s\n")

def setup_parameters(discrete:bool,
                     estimation,
                     architecture,
                     folders,
                     dataset):

    sample_size_pct = dataset.sample_size_pct
    dataset_name, _ = os.path.splitext(os.path.basename(dataset.file))
    string_arch = str(architecture.hidden_layer_sizes).strip('[]').replace(' ', '')

    if discrete:
        result_filename = "bins{}_epochs{}_arch{}_lr{}_samplepct{}.csv".format(estimation.discrete.bins,
                                                                               architecture.epochs,
                                                                               string_arch,
                                                                               architecture.learning_rate,
                                                                               sample_size_pct)
        gifs_root_path = folders.gifs.discrete    
        figures_root_path = folders.figures.discrete
        results_root_folder = folders.results.discrete
        estimation_param = estimation.discrete.bins
    else:
        result_filename = "kernelsize{}_epochs{}_arch{}_lr{}_samplepct{}.csv".format(estimation.continuos.kernel_size,
                                                                                     architecture.epochs,
                                                                                     string_arch,
                                                                                     architecture.learning_rate,
                                                                                     sample_size_pct)
        gifs_root_path = folders.gifs.continuos
        figures_root_path = folders.figures.continuos
        results_root_folder = folders.results.continuos
        estimation_param = estimation.continuos.kernel_size

    return (gifs_root_path, 
            figures_root_path, 
            results_root_folder, 
            estimation_param, 
            result_filename, 
            dataset_name, 
            sample_size_pct)

if __name__ == "__main__":
    main()
