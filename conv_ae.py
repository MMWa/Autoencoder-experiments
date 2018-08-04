import random
import time
from os import listdir

import keras
from keras.datasets import cifar10
from keras_preprocessing.image import load_img, img_to_array
from scipy import misc
import numpy as np
from keras import Input, Model
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, Conv2D
from keras.callbacks import ModelCheckpoint
from keras_tqdm import TQDMCallback
from tqdm import tqdm


class model:
    def __init__(self):
        input_img, encoder, decoder = self.__define
        self.autoencoder = Model(input_img, decoder)
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['acc'])

        self.input_img = input_img
        self.encoder = encoder
        self.decoder = decoder

    @property
    def __define(self):
        input_img = Input(shape=(None, None, 3,))

        x = Conv2D(12, (6, 6), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(12, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(24, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(48, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(48, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(24, (6, 6), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(12, (6, 6), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(12, (6, 6), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)

        decoded = Convolution2D(3, (8, 8), activation='sigmoid', padding='same')(x)

        return input_img, encoded, decoded

    def predict(self, x):
        x = keras.utils.normalize(x, axis=-1, order=2)
        y = self.autoencoder.predict(x)
        return np.multiply(y, 255)

    def test(self, filename):
        x = misc.imread(filename)
        x = np.expand_dims(x, axis=0)
        y = self.predict(x)
        misc.imsave('output/' + time.strftime("%Y%m%d-%H%M%S") + '.png', y[0, :, :, :])


def load_photos(directory, full_list, sample_size):
    images = []
    file_list = random.sample(full_list, sample_size)
    for name in tqdm(file_list):
        filename = directory + '/' + name
        image = load_img(filename, target_size=(128, 128))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = keras.utils.normalize(image, axis=-1, order=2)
        images.append(image)
    return images


if __name__ == "__main__":
    filepath = "models/conv-ae-" + time.strftime("%Y%m%d-%H%M%S") + "-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=0, period=1)

    full_list = listdir("input")
    test_file = 'test/bike_0003.png'
    model = model()

    # model.autoencoder.load_weights("models/conv-ae-01.hdf5")
    model.test(test_file)

    [x_train, _], [_, _] = cifar10.load_data()
    x_train = keras.utils.normalize(x_train, axis=-1, order=2)

    for _ in range(10):
        # x_train = np.array(load_photos("input", full_list, 1000))
        # x_train = np.squeeze(x_train, axis=1)

        model.autoencoder.fit(x=x_train, y=x_train, epochs=1, verbose=0, callbacks=[TQDMCallback(), checkpoint])
        model.test(test_file)
