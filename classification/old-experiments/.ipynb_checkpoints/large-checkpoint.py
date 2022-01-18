import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# create a data generator
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=90,
    brightness_range=[1.0, 1.75],
    zoom_range=[0.5, 1.0],
    horizontal_flip = True,
    validation_split=0.2)

# load and iterate training dataset
train_it = datagen.flow_from_directory(
    '../master-data/train_large/rgb', 
    target_size=(224, 224),
    subset="training", 
    class_mode="categorical",
    shuffle=True,
    batch_size=256)

val_it = datagen.flow_from_directory(
    '../master-data/train_large/rgb', 
    target_size=(224, 224), 
    subset="validation", 
    class_mode="categorical",
    shuffle=True,
    batch_size=256)

def plot_history(history):
    '''Plot loss and accuracy as a function of the epoch,
    for the training and validation datasets.
    '''
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Get number of epochs
    epochs = range(len(acc))

    # Plot training and validation accuracy per epoch
    plt.plot(epochs, acc, label="acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('accuracy-large.png')
    
    # Plot training and validation loss per epoch
    plt.figure()

    plt.plot(epochs, loss, label="loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('loss-large.png')
    
conv_model = VGG16(weights='imagenet', include_top=False, classes=11, input_shape=(224,224,3))

for layer in conv_model.layers:
    layer.trainable = False

# flatten the output of the convolutional part: 
x = Flatten()(conv_model.output)
# three hidden layers
x = Dense(100, activation='relu')(x)
x = Dense(100, activation='relu')(x)
x = Dense(100, activation='relu')(x)

# final softmax layer
predictions = Dense(11, activation='softmax')(x)

# creating the full model:
full_model = Model(inputs=conv_model.input, outputs=predictions)

from tensorflow.keras.optimizers import Adamax

full_model.compile(loss='categorical_crossentropy',
                  optimizer=Adamax(),
                  metrics=['acc'])

history = full_model.fit_generator(
    train_it, 
    validation_data = val_it,
    workers=14,
    epochs=100,
)

full_model.save_weights('vgg16-weights-large.h5')
full_model.save('vgg16-model-large')