from glob import glob

filenames = sorted([file for file in glob("gif_test2/*")], key=lambda x:int(x.split(".png")[0][17:]))

import imageio
with imageio.get_writer('fst_trial.gif', mode='I',duration=0.02) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)