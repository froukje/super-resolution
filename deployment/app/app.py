import os
import glob
import shutil
import uuid
import time
from datetime import datetime
from datetime import timedelta
import pytz
import requests
from flask import Flask, request, render_template, redirect
import pickle
from werkzeug.utils import secure_filename 
import numpy as np
from PIL import Image
from patchify import patchify, unpatchify

app = Flask(__name__)

app.config['IMAGE_UPLOADS'] = os.path.join(os.getcwd(), 'static')

@app.route('/', methods=['POST', 'GET'])
def home():
    ''' delete saved images older than 5 minutes'''
    delete()
    return render_template('index.html')


@app.route('/predict',methods=['POST', 'GET'])
def predict():
    """Grabs the input values and uses them to make prediction"""
    if request.method == 'POST':
        image = request.files["file"]
        if image.filename == '':
            print("Filename is invalid")
            return redirect(request.url)

        filepost = secure_filename(image.filename).split('.')[1]
        # create unique filename
        now = datetime.now(pytz.timezone('Europe/Amsterdam'))
        time = f"{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}"
        name = f"{str(uuid.uuid4())}_{time}"
        filename = f'{name}.{filepost}'
        basedir = os.path.abspath(os.path.dirname(__file__))
        path_static = os.path.join(basedir, app.config['IMAGE_UPLOADS'])
        img_path = os.path.join(path_static, 'tmp')
        img_file = os.path.join(img_path, filename)
        image.save(img_file)
        img_lr = Image.open(img_file).convert('RGB')
     
        # cut image in 96x96 patches
        # fill np array with 0s, so that width and height are divisible by 96
        img_lr_array = np.asarray(img_lr)
        if img_lr.size[0] % IMG_SIZE:
            next_div_width = img_lr.size[0] + (IMG_SIZE - img_lr.size[0] % IMG_SIZE)
        else:
            next_div_width = img_lr.size[0] 
        if img_lr.size[1] % IMG_SIZE:
            next_div_height = img_lr.size[1] + (IMG_SIZE - img_lr.size[1] % IMG_SIZE) 
        else:
            next_div_height = img_lr.size[1]
        img_lr_fill = np.zeros((next_div_height, next_div_width, img_lr_array.shape[-1])).astype(np.uint8)
        img_lr_fill[:img_lr_array.shape[0],:img_lr_array.shape[1],:] = img_lr_array.astype(np.uint8)
        patches = patchify(img_lr_fill, (IMG_SIZE, IMG_SIZE, img_lr_array.shape[-1]), step=IMG_SIZE)
        # upscaled array (with zero patches) 
        img_upscale_array = np.zeros((img_lr_fill.shape[0]*UPSCALE_FACTOR, img_lr_fill.shape[1]*UPSCALE_FACTOR, IMG_CHANNEL))

        # save image patches
        patches_upscaled = []
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patch = Image.fromarray(patches[i, j, 0, :, :, :])
                patch_path = os.path.join(img_path, f'{name}_{str(i).zfill(2)}_{str(j).zfill(2)}.{filepost}') 
                patch.save(patch_path)

                #response = requests.post("http://torchserve-mar:8080/predictions/srnet", files={'data': open(img_path, 'rb')})
                #response = requests.post("http://localhost:8080/predictions/srnet", files={'data': open(img_path, 'rb')})
                response = requests.post("http://localhost:8080/predictions/srnet", files={'data': open(patch_path, 'rb')})         
                data = np.array(response.json())
                patches_upscaled.append(data)
                size = IMG_SIZE*UPSCALE_FACTOR
                img_upscale_array[i*size:(i+1)*size, j*size:(j+1)*size,:] = data
        
        # delete possible zeros at border
        out_x = img_lr.size[1] * UPSCALE_FACTOR
        out_y = img_lr.size[0] * UPSCALE_FACTOR
        output_size = (out_x, out_y, IMG_CHANNEL)
        img_output = np.zeros(output_size)
        img_output = img_upscale_array[:out_x,:out_y,:]
        prediction = Image.fromarray((img_output*255).astype(np.uint8))
        output_name = f"tmp/{name}_out.{filepost}" 
        output_path = os.path.join(path_static, output_name)
        prediction.save(output_path)

        return render_template('index.html', output_name=output_name)
    
def delete():
    ''' delete saved images if timestamp is older than 5 minutes'''

    now = datetime.now()
    time = f"{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}"
    n = 5
    # Subtract 5 minutes from datetime object
    delete_time = now - timedelta(minutes=n)

    path = os.path.join(app.config['IMAGE_UPLOADS'], 'tmp')
    all_files = glob.glob(f'{path}/*')
    if all_files != []:
        # uuid has length of 93
        dates = [file_[93:].split('.')[0].split('_')[0] for file_ in all_files]
        time_saved = [datetime.strptime(date, '%Y-%m-%d-%H-%M') for date in dates]
        for i, file_ in enumerate(all_files):
            if time_saved[i] < delete_time:
                os.remove(file_)

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/more')
def more():
    return render_template('more.html')

if __name__ == "__main__":
    IMG_SIZE = 96
    IMG_CHANNEL = 3
    UPSCALE_FACTOR = 4
    app.run(debug=True, host="0.0.0.0", port=9696)
