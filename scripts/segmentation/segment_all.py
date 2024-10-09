import time
import traceback

import h5py
import torch

from src.data_access import ImageReader
from src.segmentation import make_mask_patch_based
from src.segmentation.model import unet

start = time.time()

print("cuda:", torch.cuda.is_available())
model = unet(
    "models/segmentation/unet/2023-03-15-unet-16x-bs16-ps256-lr0.0001/2023-03-15-unet-16x-bs16-ps256-lr0.0001-e3v49.pt"
)
file_16x = h5py.File("dataset/AD+cent.hdf5", "r")

for image_name in file_16x:
    print("segmenting", image_name, time.time() - start)
    try:
        result_file = h5py.File(f"result/images/{image_name}.hdf5", "a")
    except BlockingIOError:
        print("cannot access result file")
        continue
    if "grey-matter" in result_file:
        print("already segmented")
        continue
    try:
        hdf5_reader = ImageReader(file_16x[f"{image_name}/image_file/16x"], 1024, 1024)
        with torch.no_grad():
            grey_matter = (
                make_mask_patch_based(model, hdf5_reader)
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            )
            print("grey matter segmented", time.time() - start)
            result_file.create_dataset("grey-matter", data=grey_matter)
            print("grey matter saved", time.time() - start)
    except BaseException as e:
        print(traceback.print_exc())
        print(e)
        print("continuing")
        continue
    print()
    result_file.close()
