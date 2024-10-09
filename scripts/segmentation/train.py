import argparse
import os
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

import src.segmentation.model as m
from src.data_access import DatasetReader
from src.util import LabelEnum


def setup_cuda():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Using CPU.")
    torch.cuda.empty_cache()
    return device


DEVICE = setup_cuda()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Segmentation training")
    parser.add_argument("--name", default="test", type=str)
    parser.add_argument("-lr", "--learning_rate", default=0.001, type=float)
    parser.add_argument("-ps", "--patch_size", default=256, type=int)
    parser.add_argument("-bs", "--batch_size", default=16, type=int)
    parser.add_argument("-s", "--stride", default=-1, type=int)
    parser.add_argument("-e", "--epochs", default=10000, type=int)
    parser.add_argument("-dl", "--downsample_level", default=16, type=int)
    parser.add_argument("-mt", "--model_type", default="unet")
    parser.add_argument("-t", "--timed", action="store_true")
    parser.add_argument("-nw", "--num_workers", default=0, type=int)
    parser.add_argument("-l", "--load", default="")
    args = parser.parse_args()

    run_name = f"{args.name}-{args.model_type}-{args.downsample_level}x-bs{args.batch_size}-ps{args.patch_size}-lr{args.learning_rate}"

    print(
        "Settings:",
        "\n\tname:",
        run_name,
        "\n\tmodel:",
        args.model_type,
        f"\n\tdownsample level: {args.downsample_level}x",
        "\n\tbatch size:",
        args.batch_size,
        "\n\tpatch size:",
        args.patch_size,
        "\n\tstride:",
        args.stride,
        "\n\tlearning rate:",
        args.learning_rate,
        "\n\tnumber of epochs:",
        args.epochs,
        "\n\ttime elapsed:",
        time.time() - start_time,
    )

    return args, run_name


def train_epoch(model, data_loader, optim, criterion, info_it=None, print_times=False):
    epoch_start = time.time()
    batch_start = epoch_start
    model.train()
    iteration = timed_train_iteration if print_times else train_iteration
    running_loss = torch.zeros(1).detach().to(DEVICE)
    info_loss = torch.zeros(1).detach().to(DEVICE)
    info = []
    for i, data in enumerate(data_loader):
        model_start = time.time()
        batch_loss = iteration(model, data, optim, criterion)
        running_loss += batch_loss
        info_loss += batch_loss
        model_end = time.time()
        if info_it and (i + 1) % info_it == 0:
            avg_info_loss = info_loss.item() / info_it
            print(
                "\ttotal time",
                model_end - batch_start,
                "\n\tloading time:",
                model_start - batch_start,
                "\n\ttraining time:",
                model_end - model_start,
                "\n\tbatch time including print:",
                time.time() - batch_start,
                "\n\tavg loss:",
                avg_info_loss,
                f"\n\tprogress: {(i+1) / len(data_loader) * 100}%",
                "\n",
            )
            info.append(avg_info_loss)
            info_loss = torch.zeros(1).detach().to(DEVICE)
        batch_start = time.time()
    avg_loss = running_loss.item() / len(data_loader)
    print("train epoch done, time ", time.time() - epoch_start)
    return avg_loss, info


def train_iteration(model, data, optim, criterion):
    patches = data[0].to(DEVICE)
    labels = data[1].to(DEVICE)
    optim.zero_grad()
    out = model(patches)
    loss = criterion(out, labels)
    loss.backward()
    optim.step()
    res = loss.detach()
    return res


def timed_train_iteration(model, data, optim, criterion):
    batch_start = time.time()
    patches = data[0].to(DEVICE)
    labels = data[1].to(DEVICE)
    cuda_time = time.time()
    optim.zero_grad()
    out = model(patches)
    loss = criterion(out, labels)
    forward_time = time.time()
    loss.backward()
    optim.step()
    backward_time = time.time()
    res = loss.detach()
    detach_time = time.time()
    item_time = time.time()
    print(
        "\n\ttotal time (no print):",
        detach_time - batch_start,
        "\n\ttime to cuda:",
        cuda_time - batch_start,
        "\n\tforward:",
        forward_time - cuda_time,
        "\n\tbackward:",
        backward_time - forward_time,
        "\n\tdetach time:",
        detach_time - backward_time,
        "\n\titem time:",
        item_time - detach_time,
        "\n\tprint time:",
        time.time() - item_time,
    )
    return res


def validation_epoch(model, data_loader, criterion):
    model.eval()
    running_validation_loss = torch.zeros(1).detach().to(DEVICE)
    for i, data in enumerate(data_loader):
        batch_loss = validation_iteration(model, data, criterion)
        running_validation_loss += batch_loss
    avg_validation_loss = running_validation_loss.item() / (len(data_loader))
    return avg_validation_loss


def validation_iteration(model, data, criterion):
    patches = data[0].to(DEVICE)
    labels = data[1].to(DEVICE)
    model.zero_grad()
    out = model(patches).detach()
    loss = criterion(out, labels)
    return loss.detach()


if __name__ == "__main__":
    start_time = time.time()

    args, run_name = parse_arguments()

    ############
    # SETTINGS #
    ############
    name = args.name
    downsample_lvl = args.downsample_level
    batch_size = args.batch_size
    patch_size = args.patch_size
    stride = args.stride if args.stride > -1 else patch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    model_type = args.model_type
    num_workers = args.num_workers

    ###############
    # PREPARATION #
    ###############

    # Assumes only unet is available
    # Uses pixel-wise labels that indicate whether a pixel belongs to grey matter or not
    if model_type == "unet":
        label_type = LabelEnum.PIXEL
        model = m.unet(args.load)
    else:
        raise ValueError(f"{model_type} is not a valid model to use")

    data_file = "./dataset/linked_images_labels.hdf5"

    with h5py.File(data_file, "r") as file:
        train_reader = DatasetReader(
            file["train"],
            patch_size,
            stride,
            downsample_lvl,
            label_type,
            dtype=np.float64,
        )
        train_loader = DataLoader(
            train_reader, batch_size, shuffle=True, num_workers=num_workers
        )

        print("training dataloader ready", time.time() - start_time)
        print("\tNumber of patches:", len(train_reader))
        print("\tNumber of batches:", len(train_loader))

        validation_reader = DatasetReader(
            file["validation"],
            patch_size,
            stride,
            downsample_lvl,
            label_type,
            dtype=np.float64,
        )
        validation_loader = DataLoader(
            validation_reader, batch_size, shuffle=True, num_workers=num_workers
        )
        print("validation dataloader ready", time.time() - start_time)
        print("\tNumber of patches:", len(validation_reader))
        print("\tNumber of batches:", len(validation_loader))

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()
    print("model and optimizer ready", time.time() - start_time, flush=True)

    os.makedirs(f"./loss/{run_name}", exist_ok=True)
    os.makedirs(f"./models/segmentation/{model_type}/", exist_ok=True)

    info_it = int(np.floor(len(train_loader) / 10))
    info_loss = []
    train_loss = []
    validation_loss = []
    print("Setup done")

    ############
    # TRAINING #
    ############
    for e in range(epochs):
        # trains and validates model
        print("epoch", e)
        avg_train_loss, info = train_epoch(
            model, train_loader, optim, criterion, info_it=None, print_times=args.timed
        )
        info_loss = info_loss + info
        np.save(f"./loss/{run_name}/info-loss.npy", np.asarray(info_loss))
        train_loss.append(avg_train_loss)
        np.save(f"./loss/{run_name}/train-loss.npy", np.asarray(train_loss))
        torch.save(
            model.state_dict(), f"./models/segmentation/{model_type}/{run_name}-e{e}.pt"
        )
        print("average training loss:", avg_train_loss)

        model.eval()
        print("epoch", e, "validating")
        avg_validation_loss = validation_epoch(model, validation_loader, criterion)
        validation_loss.append(avg_validation_loss)
        np.save(f"./loss/{run_name}/validation-loss.npy", np.asarray(validation_loss))
        print("average validation loss:", avg_validation_loss)
