import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
import numpy as np
import os
root1 = "results"
root = "/home/ps/Documents/project/MMSFormer/predict"
files = os.listdir(root1)
# for i in files:
#     # res1 = Image.open(os.path.join(root1, i))
#     # res = Image.open(os.path.join(root, i))
#     # pass
#     res = tiff.imread(os.path.join(root, i))
#     res1 = tiff.imread(os.path.join(root1, i))
#     pass
#     # tiff.imsave(os.path.join(root, i), res)
    
res = Image.open(".vis/Tile_209_tile_23_19.tif")
pass