import argparse
import random

import h5py
import numpy as np
from skimage.io import imsave
from skimage.util import img_as_ubyte

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number", default=3, type=int)
    args = parser.parse_args()
    print(args)

    n = args.number

    file_16x = h5py.File("dataset/16x_set.hdf5", "r")
    keys = [key for key in file_16x["test"].keys()]
    random.shuffle(keys)
    picks = keys[:n]

    for image_name in picks:
        print(image_name)

        temp = file_16x[f"test/{image_name}/image_file/16x"][()]
        imsave(f"visuals/paper/{image_name}-image.png", np.transpose(temp, (1, 2, 0)))

        temp = file_16x[f"test/{image_name}/label_file/16x"][()]
        imsave(f"visuals/paper/{image_name}-label.png", img_as_ubyte(temp))

        result_file = h5py.File(f"result/images/{image_name}.hdf5")
        temp = result_file["grey-matter"][()]
        imsave(f"visuals/paper/{image_name}-prediction.png", img_as_ubyte(temp))
        result_file.close()

        print()

    file_16x.close()
