import os
import glob
import numpy as np
from PIL import Image, ImageFilter

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
    patch_size_small = 96
    patch_size_big = 384 # 96*4
    imgs_hr_lr = []
    for file_ in images_path:
        name = file_.split('/')[-1].split('.')[0]
        img = Image.open(file_)
        
        img_big = img.resize((patch_size_big, patch_size_big), Image.ANTIALIAS)
        img_big = img_big.convert('RGB')
        img_small = img.filter(ImageFilter.GaussianBlur)
        img_small = img_small.resize((patch_size_small, patch_size_small), Image.ANTIALIAS)
        img_small = img_small.convert('RGB')

        quality_val = 90
        img_big.save(os.path.join(path_hr, f'{name}.jpg'), quality=quality_val) 
        img_small.save(os.path.join(path_lr, f'{name}.jpg'), quality=quality_val) 

        #imgs_hr_lr.append(img_array_crop)
    #return imgs_hr_lr

def cut_and_save_images(images_hr_lr):
    patch_size_small = 96
    patch_size_big = 384 # 96*4
    for img in images_hr_lr:
        img_big = Image.fromarray(img)
        img_big = img_big.resize((patch_size_big, patch_size_big), Image.ANTIALIAS)
        img_big = img_big.filter(ImageFilter.GaussianBlur)
        img_small = img_big.resize((patch_size_small, patch_size_small), Image.ANTIALIAS)

        name = f"{str(uuid.uuid4())}" #f'{img[2]}'
        quality_val = 90
        img_big.save(os.path.join(path_hr, f'{name}.png'), quality=quality_val) 
        img_small.save(os.path.join(path_lr, f'{name}.png'), quality=quality_val) 

        

def main(): 
    images_path = read_raw_images(path_raw_images)
    print('read image paths')
    images_hr_lr = prepare_images(images_path)
    print('images prepared')

if __name__ == '__main__':
    
    path_raw_images = 'data/coco/train2017'
    path_lr = 'data/coco/LR-coco-train'
    path_hr = 'data/coco/HR-coco-train'

    main()

