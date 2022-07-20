import os
import glob
import numpy as np
from PIL import Image

def read_raw_images(path):
    '''read image pathes'''
    images_path = glob.glob(f'{path}/*')
    images_path.sort()

    return images_path

def prepare_images(images_path):
    '''
    prepare raw images
    * crop to size divisable by 4
    * downscale to lower resolution
    * return both images
    '''
    imgs_hr_lr = []
    for file_ in images_path:
        name = file_.split('/')[-1].split('.')[0]
        img = Image.open(file_)
        # crop image to a size divisable by 4
        new_height = int((img.size[0] - img.size[0]%4)) 
        new_width = int((img.size[1] - img.size[1]%4)) 
        img_array = np.asarray(img)
        img_array_crop = np.zeros((new_width, new_height, 3))
        img_array_crop = img_array[:new_width, :new_height, :]
        img_height = int(new_height/4)
        img_width = int(new_width/4)
        img = Image.fromarray(img_array_crop)
        img_down = img.resize((img_height, img_width), Image.ANTIALIAS)
        img_array_down = np.asarray(img_down)
        imgs_hr_lr.append((img_array_crop, img_array_down, name))
    return imgs_hr_lr

def cut_and_save_images(images_hr_lr):
    patch_size_small = 96
    patch_size_big = 384 # 96*4
    for img in images_hr_lr:
        img_big = Image.fromarray(img[0])
        img_small = Image.fromarray(img[1])
        img_big = img_big.resize((patch_size_big, patch_size_big), Image.ANTIALIAS)
        img_small = img_small.resize((patch_size_small, patch_size_small), Image.ANTIALIAS)

        name = f'{img[2]}'
        img_big.save(os.path.join(path_hr, f'{name}_big.png')) 
        img_small.save(os.path.join(path_lr, f'{name}_small.png')) 

        

def main(): 
    images_path = read_raw_images(path_raw_images)
    print('read image paths')
    images_hr_lr = prepare_images(images_path)
    print('images prepared')
    cut_and_save_images(images_hr_lr)
    print('images saved')

if __name__ == '__main__':
    
    path_raw_images = 'data/raw'
    path_lr = 'data/LR'
    path_hr = 'data/HR'

    main()

