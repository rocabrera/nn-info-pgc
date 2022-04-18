import os
from pathlib import Path



def crete_gif_folder(gif_root_folder:str,
                     dataset_name:str) -> str:


    folder_path = os.path.join(gif_root_folder, dataset_name)
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    return folder_path

def crete_image_folder(figure_root_folder:str,
                       dataset_name:str, 
                       result_filename:str) -> str:

    result_filename_no_ext, _ = os.path.splitext(os.path.basename(result_filename))

    folder_path = os.path.join(figure_root_folder, dataset_name, result_filename_no_ext)
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    return folder_path


def crete_result_folder(result_root_folder:str, 
                        dataset_name:str) -> str:

    folder_path = os.path.join(result_root_folder, dataset_name)
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    return folder_path

def write_header_result_experiment(result_folder:str, 
                                   result_filename:str) -> str:

    file_path = os.path.join(result_folder, result_filename)
    with open(file_path, "a") as f:
        f.write("epoch;rand_init;layer;I(X,T);I(Y,T);valid_auc;train_auc;valid_loss;train_loss\n")
    # if not os.path.exists(file_path):
    #     with open(file_path, "a") as f:
    #         f.write("epoch;rand_init;layer;I(X,T);I(Y,T);accuracy\n")
    # else:
    #     raise Exception("Experiment already done.")
    
    return file_path
