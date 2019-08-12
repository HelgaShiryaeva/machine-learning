from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.inception_v3 import InceptionV3
import matplotlib.pyplot as plt


class InceptionV3TL:
    CLASSES = 62

    EPOCHS = 5
    STEPS_PER_EPOCH = 320
    VALIDATION_STEPS = 64

    MODEL_FILE = 'inception_chars74k.model'

    def __init__(self):
        base_model = InceptionV3(weights='imagenet', include_top=False)

        x = base_model.output
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dropout(0.4)(x)
        predictions = Dense(self.CLASSES, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer='Adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self. model = model

    @staticmethod
    def plot_training(history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))

        plt.plot(epochs, acc, 'r.')
        plt.plot(epochs, val_acc, 'r')
        plt.title('Training and validation accuracy')

        plt.figure()
        plt.plot(epochs, loss, 'r.')
        plt.plot(epochs, val_loss, 'r-')
        plt.title('Training and validation loss')
        plt.show()

    def train(self, train_generator, validation_generator):
        history = self.model.fit_generator(
            train_generator,
            epochs=self.EPOCHS,
            steps_per_epoch=self.STEPS_PER_EPOCH,
            validation_data=validation_generator,
            validation_steps=self.VALIDATION_STEPS)

        self.model.save(self.MODEL_FILE)

        return history
