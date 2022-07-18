import os
import shutil
import requests
from flask import Flask, request, render_template, redirect
import pickle
from werkzeug.utils import secure_filename 
import numpy as np
from PIL import Image

app = Flask(__name__)

app.config['IMAGE_UPLOADS'] = os.path.join(os.getcwd(), 'static')

@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST', 'GET'])
def predict():
    """Grabs the input values and uses them to make prediction"""
    print(f"REQUEST METHOD: {request.method}")
    if request.method == 'POST':
        print(os.getcwd()) 
        image = request.files["file"]
        if image.filename == '':
            print("Filename is invalid")
            return redirect(request.url)

        filename = secure_filename(image.filename)

        basedir = os.path.abspath(os.path.dirname(__file__))
        img_path = os.path.join(basedir, app.config['IMAGE_UPLOADS'], filename)
        image.save(img_path)
        response = requests.post("http://torchserve-mar:8080/predictions/srnet", files={'data': open(img_path, 'rb')})
        #response = requests.post("http://localhost:8080/predictions/srnet", files={'data': open(img_path, 'rb')})

        name = filename.split('.')[0]
        output_name = f"{name}_out.jpg" 
        output_path = os.path.join(basedir, 'static', app.config['IMAGE_UPLOADS'], output_name)
        data = np.array(response.json())
        prediction = Image.fromarray((data*255).astype(np.uint8))
        prediction.save(output_path)

    return render_template('index.html', prediction_text='Upscaled Image', output_name=output_name)
    

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
