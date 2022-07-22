import os
import glob
import h5py
import time
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
from PIL import Image
from multiprocessing import Pool, Process, Manager

def read_data(image):
    
    img = Image.open(image)
    
    return img


def main(args):


    start_time = time.time()
    # low resolution images
    data_lr = []
    path = os.path.join(args.data_dir, f'LR_{args.data_name}')
    print(f'path: {path}')
    imgs_lr = glob.glob(os.path.join(path, "*"))
    imgs_lr.sort()
    print(f'Nr of images low-res: {len(imgs_lr)}')
    print(imgs_lr[:5])
    
    pool = Pool(8)
    
    data_lr = pool.map(read_data, imgs_lr)
    print(f'data read ({time.time() - start_time:.2f} sec)')

    data_array_lr = np.stack(data_lr)
    print(f'LR data shape: {data_array_lr.shape}')
    
    # high resolution images
    data_hr = []
    path = os.path.join(args.data_dir, f'HR_{args.data_name}')
    print(f'path: {path}')
    imgs_hr = glob.glob(os.path.join(path, "*"))
    imgs_hr.sort()
    print(f'Nr of images high-res: {len(imgs_hr)}')
    print(imgs_hr[:5])
    data_hr = pool.map(read_data, imgs_hr)

    data_array_hr = np.stack(data_hr)
    print(f'HR data shape: {data_array_hr.shape}')
   
    pool.close()

    # train test split
    lr_train, lr_valid, hr_train, hr_valid = train_test_split(data_array_lr, data_array_hr, random_state=42)
    
    data_list = [[lr_train, hr_train], [lr_valid, hr_valid]]
    name_list = [f'train_{args.data_name}', f'valid_{args.data_name}']
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
    parser.add_argument('--data-name', type=str, default='feli')
    args = parser.parse_args()

    for key, value in vars(args).items():
        print(f'{key}: {value}')

    main(args)
