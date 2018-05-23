# ID=bba

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from os import listdir
import numpy as np

class cnn:
    '''
        class implementing keras implementation of convolution neural network
    '''

    def __init__(self, x_shape, y_shape, epochs, batch_size):
        '''
        initialize cnn class

        :param x_shape: height of image
        :param y_shape: width of image
        :param epochs: number of epoch
        :param batch_size: size of batch
        '''
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.epochs = epochs
        self.batch_size = batch_size

    def set_data(self, train_file, validation_file):
        '''
        set data
        :param train_file: path to train file
        :param validation_file: path to validation file
        '''
        self.train_file = train_file
        self.validation_file = validation_file

    def set_model(self):
        '''
        initialize cnn model
        :return: model
        '''
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(self.x_shape, self.y_shape, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten(input_shape=model.output_shape[1:]))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(4, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model = model

    def fit(self):
        '''
        train model
        :return: path to file with weights.
        '''
        train_datagen = ImageDataGenerator(
            rotation_range=0,
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        validation_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            self.train_file,
            target_size=(self.x_shape, self.y_shape),
            batch_size=self.batch_size,
            class_mode='categorical'
        )

        validation_generator = validation_datagen.flow_from_directory(
            self.validation_file,
            target_size=(self.x_shape, self.y_shape),
            batch_size=self.batch_size,
            class_mode='categorical')

        self.model.fit_generator(
            train_generator,
            steps_per_epoch=2000 // self.batch_size,
            epochs=self.epochs,
            validation_data=validation_generator,
            validation_steps=800 // self.batch_size
        )

        self.model.save_weights('first_try.h5')
        print(train_generator.class_indices)
        return 'weights.h5'


    def predict(self, weights_file, test_file):
        '''
        predict the classes for new images
        :param weights_file: path to file with weights
        :param test_file: path to file with test images
        :return: array with classes
        '''
        self.model.load_weights(weights_file)
        results = []
        for file in listdir(test_file):
            img = load_img(test_file+file, target_size = (self.x_shape, self.y_shape))
            img = img_to_array(img)
            img = img / 255
            img = np.expand_dims(img, axis=0)
            results.append(self.model.predict_classes(img, batch_size=self.batch_size)[0])
        return results

    def save_to_disk(self, predictions):
        '''
        save classes to file on disk
        :param predictions: array with classes
        '''
        f = open('results.txt', 'w')
        for el in predictions:
            f.write(str(el) + '\n')
        f.close()

if __name__ == "__main__":
    cnn = cnn(150, 150, 15, 16)
    cnn.set_data('yeast_images/train', 'yeast_images/validation')
    cnn.set_model()
    weight_file = cnn.fit()
    predictions = cnn.predict(weight_file, 'yeast_images/test/')
    cnn.save_to_disk(predictions)

