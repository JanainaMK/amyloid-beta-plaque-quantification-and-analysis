import time

import h5py
import numpy as np
import torch
from torch import nn

from src.data_access import ImageReader, VsiIterator

SOFTMAX = nn.Softmax(1)


def grey_matter(model: nn.Module, vsi_reader: VsiIterator):
    x, y = vsi_reader.patch_it.shape
    dim = x + 1, y + 1
    bs = vsi_reader.batch_size
    res = np.zeros(dim)
    start = time.time()
    row_count = 0
    for patch in vsi_reader:
        r, c = vsi_reader.get_current_patch_coords()
        if r > row_count:
            print("row", row_count, "took", time.time() - start, "seconds")
            np.save("./visuals/test.npy", res)
            start = time.time()
            row_count += 1
        input = torch.Tensor(patch).double().to("cuda").detach()
        out = model(input)
        # print(r, c)
        low = c - bs
        if low < 0:
            res[r - 1, low:] = out[:-low, 1].cpu().detach().numpy()
            res[r, : bs + low] = out[-low:, 1].cpu().detach().numpy()
        else:
            res[r, low:c] = out[:, 1].cpu().detach().numpy()
        # print(out[:, 1].item())


def grey_matter_v2(model: nn.Module, image_reader: ImageReader):
    model.train(False)
    x, y = image_reader.patch_it.shape
    dim = x + 1, y + 1
    res = np.zeros(dim)
    start = time.time()
    row_count = 0
    for i in range(len(image_reader)):
        patch = image_reader[i]
        r = int(np.floor(i / dim[1]))
        c = i % dim[1]
        if r > row_count:
            print("row", row_count, "took", time.time() - start, "seconds")
            np.save("./visuals/0x.npy", res)
            start = time.time()
            row_count += 1
        input = torch.unsqueeze(torch.Tensor(patch).double().to("cuda").detach(), 0)
        out = model(input).squeeze()
        out = SOFTMAX(out)
        res[r, c] = out[1].mean().cpu().detach().numpy()


def grey_matter_unet(model: nn.Module, image_reader: ImageReader):
    model.train(False)
    x, y = image_reader.patch_it.shape
    dim = x, y
    res = np.zeros(dim)
    start = time.time()
    row_count = 0
    for i in range(len(image_reader)):
        r = int(np.floor(i / dim[1]))
        c = i % dim[1]
        if r > row_count:
            print("row", row_count, "took", time.time() - start, "seconds")
            np.save("./visuals/unet-lr0.001-pt2.npy", res)
            start = time.time()
            row_count += 1
        patch = image_reader[i]
        input = torch.unsqueeze(torch.Tensor(patch).double().to("cuda").detach(), 0)
        out = model(input)
        res[r, c] = torch.mean(out).cpu().detach().numpy()


m = torch.hub.load(
    "mateuszbuda/brain-segmentation-pytorch",
    "unet",
    in_channels=3,
    out_channels=1,
    init_features=32,
    pretrained=False,
)
model_name = "lr_test-unet-16x-bs16-ps256-lr0.001"
m.load_state_dict(torch.load(f"./models/{model_name}/{model_name}-e0.pt"))
m.double()
m.to("cuda")
m.eval()
print("model ready")

file = h5py.File("./dataset/images/Image_2015-090_temppole_BA4.hdf5", "r")
image = ImageReader(file["16x"], 256, 128)
print("data ready")

print(image.patch_it.shape)
grey_matter_unet(m, image)
