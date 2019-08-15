
from mobilenet.model import MobileNetV2C74k
from keras.applications.mobilenet_v2 import preprocess_input
from utils import preprocessing
from utils import train

TRAIN_DIR = 'D:\\machine-learning\\chars74k\\EnglishImg\\Train'
VAL_DIR = 'D:\\machine-learning\\chars74k\\EnglishImg\\Val'
WIDTH = 224
HEIGHT = 224
BATCH_SIZE = 32

model = MobileNetV2C74k()

train_generator = preprocessing.get_data_generator(path=TRAIN_DIR,
                                        preprocess_input=preprocess_input,
                                        img_height=HEIGHT,
                                        img_width=WIDTH,
                                        batch_size=BATCH_SIZE)

validation_generator = preprocessing.get_data_generator(path=VAL_DIR,
                                        preprocess_input=preprocess_input,
                                        img_height=HEIGHT,
                                        img_width=WIDTH,
                                        batch_size=BATCH_SIZE)

history = model.train(train_generator, validation_generator)

train.plot_training(history)
