import time
import os
import argparse
import numpy as np
import h5py
from src.data_access import VsiReader
import bioformats
import src.util.jvm as jvm
from split_data import list_files

images_target_dir = 'images'
def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Downsample image')
    parser.add_argument('-ps', '--patch_size', default=256, type=int)
    parser.add_argument('-s', '--stride', default=-1, type=int)
    parser.add_argument('-dl', '--downsample_level', default=16, type=int)
    parser.add_argument('-sd', '--save_dir', default='dataset', type=str)
    parser.add_argument('-id', '--images_dir', default=os.path.join('dataset', 'original', 'Abeta images 100+'), type=str)
    parser.add_argument('-rc', '--replace_created', action='store_true', default=True)
    args = parser.parse_args()
    
    print(
        'Settings:',
        f'\n\tdownsample level: {args.downsample_level}x',
        '\n\tpatch size:', args.patch_size,
        '\n\tstride:', args.stride,
        '\n\save directory:', args.save_dir,
        '\n\images directory:', args.images_dir
    )

    return args

def save_downsampled_images(args):
    """
    Save downsampled images.
    """
    downsample_lvl = args.downsample_level
    patch_size = args.patch_size
    stride = args.stride if args.stride > -1 else patch_size
    replace_created = args.replace_created
    images_dir = args.images_dir
    
    image_files = list_files(images_dir, 'vsi')
    
    print(f'\nsaving {downsample_lvl}x downsampled images')
    
    images_downsampled_dir = os.path.join(args.save_dir, 'processed', 'images')
    os.makedirs(images_downsampled_dir, exist_ok=True)
    
    try:
        jvm.start()
        for i, image_file in enumerate(image_files):
            image_name = os.path.splitext(image_file)[0].split(os.path.sep)[-1]
            
            images_downsampled_file_path = os.path.join(images_downsampled_dir, f'{image_name}.hdf5')
            if replace_created or not os.path.exists(images_downsampled_file_path):
                print(f'{i+1}/{len(image_files)} files, {image_name}')
                
                # Save image downsampled
                with h5py.File(images_downsampled_file_path, 'w') as image_downsampled_file:
                    raw_reader = bioformats.ImageReader(image_file)
                    vsi_reader = VsiReader(raw_reader, patch_size, stride , downsample_lvl, np.uint8, False, True, 13)
                    image = vsi_reader.get_full()
                    image = np.transpose(image, (2, 0, 1))
                    
                    image_downsampled_file.create_dataset(f'{downsample_lvl}x', data=image)
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        jvm.stop()
    
if __name__ == "__main__":
    args = parse_arguments()
    
    save_downsampled_images(args)
    