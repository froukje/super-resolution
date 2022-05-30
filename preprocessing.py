import os
import glob
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
from PIL import Image

def main(args):

    # low resolution images
    data_lr = []
    path = os.path.join(args.data_dir, 'LR')
    print(f'path: {path}')
    for image in glob.glob(os.path.join(path, '*.png')):
        img = Image.open(image)
        # assumes all images in each folder have the same size!
        data_lr.append(img)
    data_array_lr = np.stack(data_lr)
    print(f'LR data shape: {data_array_lr.shape}')
    
    # high resolution images
    data_hr = []
    path = os.path.join(args.data_dir, 'HR')
    print(f'path: {path}')
    for image in glob.glob(os.path.join(path, '*.png')):
        img = Image.open(image)
        # assumes all images in each folder have the same size!
        data_hr.append(img)
    data_array_hr = np.stack(data_hr)
    print(f'HR data shape: {data_array_hr.shape}')
 
    # train test split
    lr_train, lr_valid, hr_train, hr_valid = train_test_split(data_array_lr, data_array_hr, random_state=42)
    
    data_list = [[lr_train, hr_train], [lr_valid, hr_valid]]
    name_list = ['train', 'valid']
    # save data: lr data is saved as X, hr as y 
    for name, data in zip(name_list, data_list):
        h5_file = h5py.File(os.path.join(args.data_dir, f'{name}.h5'), 'w')
        chunks = (1,) + data[0].shape[1:]
        h5_file.create_dataset('X', data=data[0], shape=data[0].shape, chunks=chunks)
        chunks = (1,) + data[1].shape[1:]
        h5_file.create_dataset('y', data=data[1], shape=data[1].shape, chunks=chunks)
        print(f'saved {name} as h5')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    args = parser.parse_args()

    for key, value in vars(args).items():
        print(f'{key}: {value}')

    main(args)
