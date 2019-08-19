from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.mobilenet_v2 import MobileNetV2
import matplotlib.pyplot as plt


class MobileNetV2C74k:

    CLASSES = 36
    EPOCHS = 55
    STEPS_PER_EPOCH = 320
    VALIDATION_STEPS = 64

    MODEL_FILE = 'mobilenet_chars74k.model'

    def __init__(self):
        base_model = MobileNetV2(weights='imagenet', include_top=False)

        x = base_model.output
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dropout(0.3)(x)
        x = Dense(1024,activation='relu')(x)
        x = Dense(512,activation='relu')(x)
        predictions = Dense(self.CLASSES, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)

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
