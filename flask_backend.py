#Flask backend to server Number_ID_App
# tutorial from https://www.youtube.com/watch?v=f6Bf3gl4hWY
#h5 and json file

from flask import Flask, render_template, request
            #render_templates- generating html from python is hard
            #this helps us define/render an html file. index.html
            #request- handles get, post, set requests to server
from scipy.misc import imsave, imread, imresize
import numpy as np      #for matrix math
import keras.models     # to import keras models
import re               #for handling large string data
import sys
import os               # for operating system data
#import base64
import codecs

sys.path.append(os.path.abspath('./model')) #tells app where saved model is
from load import *

# !!! Comment out these 4 lines when serving on the cloud. for local use only
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

#init flask app
app = Flask(__name__)

global model, graph
            #Declared 2 global variables
            #model is the object for incapsulating the model file
            #graph is the computation graph- a session from inside the models
model, graph = init() #initial them to

#Helper functions

def convertImage(imgData1):   #decodes info into binary imgData
    imgstr = re.search(b'base64,(.*)',imgData1).group(1)
    with open('out.png', 'wb') as output:  #change wb to w per youtube comment
        output.write(codecs.decode(imgstr,'base64'))


@app.route('/') #tells app what happens when a user goes to a certain address
                #if user goes to "/" that is the main page.  If "/contact"...
def index():
    return render_template("index.html")  #from Flask import dependencies

@app.route('/predict/', methods=['GET', 'POST'])  #user hits "submit" button
def predict():
    imgData = request.get_data() #gets raw data from user imput image, then #reshape the image (preprocessing for model)
    convertImage(imgData)
    x = imread('out.png', mode = 'L')
    x = np.invert(x)            #bitwise inversion to make image easier to class
    x = imresize(x, (28, 28))     # resize to what we trained it on
    x = x.reshape(1, 28, 28, 1) # reshape into a 4D tensor
    print("debug2")
    with graph.as_default():
        out = model.predict(x)
        print(out)
        print(np.argmax(out,axis=1))

        response = np.array_str(np.argmax(out, axis = 1))
        return response


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
