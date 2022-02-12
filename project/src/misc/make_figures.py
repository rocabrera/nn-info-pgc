import os
import imageio
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ieee')
import seaborn as sns
from tqdm import tqdm

def set_legend(ax, epoch):

    ax.set_xlabel("I(X,T)", fontsize=12)
    ax.set_ylabel("I(Y,T)", fontsize=12)
    ax.set_title(f"Época: {epoch}", fontsize=14)

    leg = ax.legend(loc='lower right', 
                    title='Layers',
                    fontsize=12)
    # make opaque legend
    for lh in leg.legendHandles:
        fc_arr = lh.get_fc().copy()
        fc_arr[:, -1] = 1
        lh.set_fc(fc_arr)
        lh.set_alpha(1)

    return leg

def create_figures(result_file_path:str, 
                   epochs:str, 
                   save_folderpath:str,
                   **kwargs:dict) -> None:

    df = pd.read_csv(result_file_path, sep=";")

    xmax, xmin = df["I(X,T)"].max(), df["I(X,T)"].min() 
    ymax, ymin = df["I(Y,T)"].max(), df["I(Y,T)"].min() 

    for epoch in tqdm(range(epochs), desc="Image Epoch"):
        _, ax = plt.subplots(figsize = (6,4))

        g = sns.scatterplot(data=df.query(f"epoch=={epoch}"), 
                            x='I(X,T)', 
                            y="I(Y,T)", 
                            hue="layer",
                            ax=ax,
                            **kwargs)

        plt.title(f"Época: {epoch}")
        _ = set_legend(g,epoch)

        ax.set_xlim([xmin-0.2, xmax+0.2])
        ax.set_ylim([ymin-0.2, ymax+0.2])
        plt.tight_layout()
        plt.savefig(os.path.join(save_folderpath, f"{epoch}.png"), facecolor='w')
        plt.close()


def create_gif(image_folder_path:str, gif_root_path:str, result_filename:str):

    result_filename_no_ext, _ = os.path.splitext(os.path.basename(result_filename))
    save_path = os.path.join(gif_root_path, result_filename_no_ext+".gif")


    filenames = sorted([file for file in glob(f"{image_folder_path}/*")], 
                        key=lambda x:int(os.path.basename(x).split(".png")[0]))
                        
    with imageio.get_writer(save_path, mode='I',duration=0.02) as writer:
        for filename in tqdm(filenames, desc="Gif"):
            image = imageio.imread(filename)
            writer.append_data(image)