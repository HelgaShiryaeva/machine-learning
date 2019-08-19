from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.inception_v3 import InceptionV3


class InceptionV3TL:
    CLASSES = 36

    EPOCHS = 5
    STEPS_PER_EPOCH = 320
    VALIDATION_STEPS = 64

    MODEL_FILE = 'inception_chars74k.model'

    def __init__(self):
        base_model = InceptionV3(weights='imagenet', include_top=False)

        x = base_model.output
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dropout(0.4)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(self.CLASSES, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer='Adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self. model = model


    def train(self, train_generator, validation_generator):
        history = self.model.fit_generator(
            train_generator,
            epochs=self.EPOCHS,
            steps_per_epoch=self.STEPS_PER_EPOCH,
            validation_data=validation_generator,
            validation_steps=self.VALIDATION_STEPS)

        self.model.save(self.MODEL_FILE)

        return history
