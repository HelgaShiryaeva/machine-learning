from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


class DataGenerator:

    def __init__(self):
        self.WIDTH = 299
        self.HEIGHT = 299
        self.BATCH_SIZE = 32

        self.TRAIN_DIR = 'C:\\Users\\helga_sh\\PycharmProjects\\asl-alphabet\\machine-learning\\chars74k\\EnglishImg\\Train'
        self.TEST_DIR = 'C:\\Users\\helga_sh\\PycharmProjects\\asl-alphabet\\machine-learning\\chars74k\\EnglishImg\\Test'

    def test_train_generators(self):
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        validation_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        train_generator = train_datagen.flow_from_directory(
            self.TRAIN_DIR,
            target_size=(self.HEIGHT, self.WIDTH),
            batch_size=self.BATCH_SIZE,
            class_mode='categorical')

        validation_generator = validation_datagen.flow_from_directory(
            self.TEST_DIR,
            target_size=(self.HEIGHT, self.WIDTH),
            batch_size=self.BATCH_SIZE,
            class_mode='categorical')

        return train_generator, validation_generator

    @staticmethod
    def plot_random_samples(x_batch, y_batch):
        plt.figure(figsize=(12, 9))
        for k, (img, lbl) in enumerate(zip(x_batch, y_batch)):
            plt.subplot(4, 8, k + 1)
            plt.imshow((img + 1) / 2)
            plt.axis('off')
        plt.show()
