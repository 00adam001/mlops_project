import keras
from keras.models import Model
from keras.layers import Activation, Conv2D, BatchNormalization, Dense, Flatten, Reshape, Input

def get_model() -> Model:
    model = keras.models.Sequential()
    model.add(Input(shape=(9, 9, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')) 
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(81 * 9))
    model.add(Reshape((-1, 9)))
    model.add(Activation('softmax'))
    return model
