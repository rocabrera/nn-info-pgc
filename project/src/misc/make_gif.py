import imageio
import os
from glob import glob
from pathlib import Path


root_path = os.getcwd()
folder_type = "discrete"
dataset = "make_circles_3"
folder_exp = "bins10_epochs1000_arch5_lr0.1"

images_path = os.path.join(root_path,"images", folder_type, dataset, folder_exp)

folder_save_path = os.path.join(root_path, "gifs", folder_type, dataset)
Path(folder_save_path).mkdir(parents=True, exist_ok=True)

save_path = os.path.join(folder_save_path, folder_exp+".gif")


filenames = sorted([file for file in glob(f"{images_path}/*")], 
                    key=lambda x:int(os.path.basename(x).split(".png")[0]))
                    
with imageio.get_writer(save_path, mode='I',duration=0.02) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)