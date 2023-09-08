import time
import torch
import numpy as np

from src.segmentation.model import save_model_from_name

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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
        if info_it and (i+1) % info_it == 0:
            avg_info_loss = info_loss.item() / info_it
            print(
                '\ttotal time', model_end - batch_start,
                '\n\tloading time:', model_start - batch_start,
                '\n\ttraining time:', model_end - model_start,
                '\n\tbatch time including print:', time.time() - batch_start,
                '\n\tavg loss:', avg_info_loss,
                f'\n\tprogress: {(i+1) / len(data_loader) * 100}%',
                '\n',
            )
            info.append(avg_info_loss)
            info_loss = torch.zeros(1).detach().to(DEVICE)
        batch_start = time.time()
    avg_loss = running_loss.item() / len(data_loader)
    print('train epoch done', time.time() - epoch_start)
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
    item = loss.item()
    item_time = time.time()
    print(
        '\n\ttotal time (no print):', detach_time - batch_start,
        '\n\ttime to cuda:', cuda_time - batch_start,
        '\n\tforward:', forward_time - cuda_time,
        '\n\tbackward:', backward_time - forward_time,
        '\n\tdetach time:', detach_time - backward_time,
        '\n\titem time:', item_time - detach_time,
        '\n\tprint time:', time.time() - item_time,
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


def train_partial_epoch(model, loader, optim, criterion, num_samples):
    model.train()
    running_loss = torch.zeros(1, requires_grad=False).detach().to(DEVICE)
    for i in range(num_samples):
        batch_loss = train_iteration(model, next(loader), optim, criterion)
        running_loss += batch_loss
    return running_loss.item() / num_samples


def validate_partial_epoch(model, loader, criterion, num_samples):
    model.eval()
    running_loss = torch.zeros(1, requires_grad=False).detach().to(DEVICE)
    for j in range(num_samples):
        batch_loss = validation_iteration(model, next(loader), criterion)
        running_loss += batch_loss
    return running_loss.item() / num_samples


def train_epoch_interim_validation(
        model,
        train_loader,
        validation_loader,
        optim,
        criterion,
        val_per_epoch,
        model_name,
        epoch,
):
    train_interval = int(np.floor(len(train_loader) / val_per_epoch))
    validation_interval = int(np.floor(len(validation_loader) / val_per_epoch))
    train_data = iter(train_loader)
    validation_data = iter(validation_loader)
    train_loss = np.zeros(val_per_epoch)
    validation_loss = np.zeros(val_per_epoch)

    for k in range(val_per_epoch):
        # Training
        train_loss[k] = train_partial_epoch(model, train_loader, optim, criterion, )
        save_model_from_name(model, model_name, f'e{epoch}v{k}')
        np.save(f'./loss/{model_name}/{model_name}-train-loss-e{epoch}.npy', train_loss)

        # Validation
        model.eval()
        running_loss = torch.zeros(1).detach().to(DEVICE)
        for j in range(validation_interval):
            batch_loss = validation_iteration(model, next(validation_data), criterion)
            running_loss += batch_loss
        validation_loss[k] = running_loss.item() / validation_interval
        np.save(f'./loss/{model_name}/{model_name}-validation-loss-e{epoch}.npy', validation_loss)

    return train_loss, validation_loss










