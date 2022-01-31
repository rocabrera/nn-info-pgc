import imageio
import os
from glob import glob

root_path = os.getcwd()
folder_type = "discrete"
dataset = "breast_cancer"
folder_exp = "bins10_epochs20_arch5,3_lr0.1"

images_path = os.path.join(root_path,"images", folder_type, dataset, folder_exp)
save_path = os.path.join(root_path, "gifs", folder_type, dataset, folder_exp+".gif")


filenames = sorted([file for file in glob(f"{images_path}/*")], 
                    key=lambda x:int(os.path.basename(x).split(".png")[0]))
                    
with imageio.get_writer(save_path, mode='I',duration=0.02) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)