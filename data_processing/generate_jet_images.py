import numpy as np
import pathlib
import h5py
from energyflow.utils import pixelate
from tqdm import tqdm
from tools import get_data

path = pathlib.Path.cwd()



def generate_jet_images(bbx):
    # Number of pixels in the image
    npix=32
    nb_chan = 1
    
    # Image width (0.8 corresponds to -0.4 to +0.4)
    img_width=0.8
    
    # Load sample data
    X, y = get_data(bbx)
    
    # h5 file to save results
    h5_file = path.parent / "data" / "jet_images" / f"jet_images_chan_{nb_chan}-{bbx}.h5"
    if not h5_file.exists():
        hf = h5py.File(h5_file, "w")
        images = np.asarray([pixelate(x, npix=npix, img_width=img_width, nb_chan=nb_chan, norm=False) for x in X])
        hf.create_dataset("features", data=images)
        hf.create_dataset("targets", data=y)
        hf.close()

if __name__ == "__main__":
    generate_jet_images(bbx)
