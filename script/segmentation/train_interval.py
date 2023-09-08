import time
import os
import argparse
import sys

import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader

from src.data_access import DatasetReader
from src.util import LabelEnum
import src.segmentation.training as training
import src.segmentation.model as m

start_time = time.time()

########
# CUDA #
########
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()
print('cuda availible:', torch.cuda.is_available())
os.system('sstat -j $SLURM_JOBID --format JOBID,MaxRSS')

####################
# ARGUMENT PARSING #
####################
parser = argparse.ArgumentParser(description='Segmentation training')
parser.add_argument('name')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
parser.add_argument('-ps', '--patch_size', default=256, type=int)
parser.add_argument('-bs', '--batch_size', default=16, type=int)
parser.add_argument('-s', '--stride', default=-1, type=int)
parser.add_argument('-e', '--epochs', default=10000, type=int)
parser.add_argument('-dl', '--downsample_level', default=16, type=int)
parser.add_argument('-m', '--model', default='unet')
parser.add_argument('-p', '--prep', action='store_true')
parser.add_argument('-ve', '--validations_per_epoch', default=50, type=int)
parser.add_argument('-l', '--load', default='', type=str)
args = parser.parse_args()

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
val_per_epoch = args.validations_per_epoch
use_model = args.model


run_name = f'{name}-{use_model}-{downsample_lvl}x-bs{batch_size}-ps{patch_size}-lr{learning_rate}'
print(
    'Settings:',
    '\n\tname:', run_name,
    '\n\tmodel:', use_model,
    f'\n\tdownsample level: {downsample_lvl}x',
    '\n\tbatch size:', batch_size,
    '\n\tpatch size:', patch_size,
    '\n\tstride:', stride,
    '\n\tlearning rate:', learning_rate,
    '\n\tnumber of epochs:', epochs,
    '\n\tvalidations per epoch:', val_per_epoch,
    '\n\ttime elapsed:', time.time() - start_time,
)

###############
# PREPARATION #
###############
if use_model == 'unet':
    label_type = LabelEnum.PIXEL
else:
    raise ValueError(f'{use_model} is (currently) not a valid model to use')

data_file = './dataset/16x_set.hdf5'
file = h5py.File(data_file, 'r')
train_reader = DatasetReader(file['train'], patch_size, stride, downsample_lvl, label_type, dtype=np.float64)
train_loader = DataLoader(train_reader, batch_size, shuffle=True, num_workers=6)
train_interval = int(np.floor(len(train_loader) / val_per_epoch))
print('training dataloader ready', time.time() - start_time)
print('\tNumber of patches:', len(train_reader))
print('\tNumber of batches:', len(train_loader))
print('\tInterval size:', train_interval)

validation_reader = DatasetReader(file['validation'], patch_size, stride, downsample_lvl, label_type, dtype=np.float64)
validation_loader = DataLoader(validation_reader, batch_size, shuffle=True, num_workers=6)
validation_interval = int(np.floor(len(validation_loader) / val_per_epoch))
print('validation dataloader ready', time.time() - start_time)
print('\tNumber of patches:', len(validation_reader))
print('\tNumber of batches:', len(validation_loader))
print('\tInterval size:', validation_interval)

if use_model == 'unet':
    model = m.unet(args.load)
else:
    raise ValueError(f'{use_model} is not a valid model to use')
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.BCELoss()
print('model and optimizer ready',  time.time() - start_time, flush=True)

os.makedirs(f'./loss/{run_name}', exist_ok=True)
os.makedirs(f'./models/{run_name}', exist_ok=True)
train_loss = np.zeros(epochs * val_per_epoch)
validation_loss = np.zeros(epochs * val_per_epoch)
print('Setup done')
os.system('sstat -j $SLURM_JOBID --format JOBID,MaxRSS')

if args.prep:
    sys.exit()
############
# TRAINING #
############
for e in range(epochs):
    print('epoch', e)
    train_it = iter(train_loader)
    validation_it = iter(validation_loader)
    for k in range(val_per_epoch):
        print('training')
        start_time = time.time()
        avg_training_loss = training.train_partial_epoch(model, train_it, optim, criterion, train_interval)
        train_loss[e * val_per_epoch + k] = avg_training_loss
        m.save_model_from_name(model, run_name, f'e{e}v{k}')
        np.save(f'./loss/{run_name}/{run_name}-train-loss.npy', train_loss)
        print('training epoch', e, 'section', k, 'done')
        print('avg training loss:', avg_training_loss, 'time:', time.time() - start_time)
        os.system('sstat -j $SLURM_JOBID --format JOBID,MAXRSS')

        print('validating')
        start_time = time.time()
        avg_validation_loss = training.validate_partial_epoch(model, validation_it, criterion, validation_interval)
        validation_loss[e * val_per_epoch + k] = avg_validation_loss
        np.save(f'./loss/{run_name}/{run_name}-validation-loss.npy', validation_loss)
        print('validating epoch', e, 'section', k, 'done')
        print('avg validation loss:', avg_validation_loss, 'time:', time.time() - start_time)
        os.system('sstat -j $SLURM_JOBID --format JOBID,MaxRSS')
        print('')

###########
# CLOSING #
###########
file.close()
