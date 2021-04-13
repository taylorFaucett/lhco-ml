import numpy as np
import pathlib
import h5py
from energyflow.utils import pixelate
from tqdm import tqdm

path = pathlib.Path.cwd()



def generate_jet_images():
    # Number of pixels in the image
    npix=32
    
    # Image width (0.8 corresponds to -0.4 to +0.4)
    img_width=0.8
    
    # Load sample data
    X = np.load(path.parent / "data" / "X.npy")
    y = np.load(path.parent / "data" / "y.npy")
    for nb_chan in (1,2):
        # h5 file to save results
        hf = h5py.File(path.parent / "data" / f"jet_images_chan_{nb_chan}.h5", "w")
        images = np.asarray([pixelate(x, npix=npix, img_width=img_width, nb_chan=nb_chan, norm=False) for x in X])
        hf.create_dataset("features", data=images)
        hf.create_dataset("targets", data=y)
    hf.close()
            




if __name__ == "__main__":
    generate_jet_images()
