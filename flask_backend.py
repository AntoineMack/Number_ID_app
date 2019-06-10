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

sys.path.append(os.path.abspath('./model')) #saves model into model folder
from load import *

#init flask app
app = Flask(__name__)

global model, graph
            #Declared 2 global variables
            #model is the object for incapsulating the model file
            #graph is the computation graph- a session from inside the models
model, graph = init() #initial them to

#Helper functions

def convertImage(imgData)   #decodes info into binary imgData
    imgstr = re.search(r'base64,(.*'.imgData1).group(1))
    with open('output.png', 'wb') as output:
        output.write(imgstr.decode('base64'))


@app.route('/') #tells app what happens when a user goes to a certain address
                #if user goes to "/" that is the main page.  If "/contact"...
def index():
    return render_template('index.html')  #from Flask import dependencies

@app.route('/predict', methods=['GET', 'POST'])  #user hits "submit" button
def predict():
    imgData = request.get_data() #gets raw data from user imput image, then
                                 #reshape the image (preprocessing for model)
    convertImage(imgData)
    x = imread('out.png', mode = 'L')
    x = np.invert(x)            #bitwise inversion to make image easier to class
    x = imresize(x, 28, 28)     # resize to what we trained it on
    x = x.reshape(1, 28, 28, 1) # reshape into a 4D tensor
    with graph.as_default():
        out = model.predict(x)
        response = np.array_str(np.argmax(out, axis = 1))
        return response


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8888)) # define port
    app.run(host='0.0.0.0', port = port)
