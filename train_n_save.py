#Trains a simple convnet on the MNIST dataset. Get to 99.25% accuracy
#after 12 epochs.  There is still margin for param tuning

from __future__ import print_function
import keras
from keras.datasets import mnist # dataset of handwritten characters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D #grabs the most important parts
                                              # of the convertImage
from keras import backend as K

batch_size = 128
num_classes = 10  #numbers 0 thru 9
epochs = 12

img_rows, img_cols = 28, 28 #input image dimensions

(X_train, y_train), (X_test, y_test) = mnist.load_data()

#3D data can have 2 forms "channels_last" (conv_dim1, conv_dim2, conv_dim3,
# channels) while "channels_first" (channels, conv_dim1, conv_dim2, conv_dim3)
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0],1 img_rows, img_cols)
    input_shape = (img_rows, img_cols, 1)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astpye('float32') #more reshaping for the NN
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes) #convert class vectors to
y_test = keras.utils.to_categorical(y_test, num_classes)   #binary class matrices

model = Sequential # BUILD!!!
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape= input_shape))
model.add(Conv2D(64, (3, 3), activation ='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  #select best features in image
model.add(Dropout(.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer= keras.optimizers.Adadela(),
                metrics= ['accuracy'])

model.fit(X_train, y_train, batch_size= batch_size, epochs = epochs,
            verbose =1, validation = (X_test, y_test)) #  TRAIN!!!!


score = model.evaluate(X_test, y_test, verbose = 0) #EVALUATE!!!
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_json = model.to_json()
with open('model.json', "w") as json_file:
    json_file.rite(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
