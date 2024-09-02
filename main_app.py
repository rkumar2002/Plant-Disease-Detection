import cv2
import requests 
import imutils
import numpy as np
from flask import Flask, render_template, Response
import pdb
import json 
from ultralytics import YOLO 
import os

model = YOLO('predicter.pt')
app = Flask(__name__)
url = "http://192.168.137.254:8080/shot.jpg"

def camera_stream():
    i=1
    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=1000, height=1800)
        result = model.predict(img,project='Temp',name='Photos',save=True,conf=0.7)
        output_folder = f'Temp/Photos{i}/'
        
        predicted_image_path = os.path.join(output_folder, "image0.jpg")
        if os.path.exists(predicted_image_path):
            img2 = cv2.imread(predicted_image_path)
            _, buffer = cv2.imencode('.jpg', img2)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            i+=1
        else:
            _, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            i+=1
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(camera_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True,port=5000)
