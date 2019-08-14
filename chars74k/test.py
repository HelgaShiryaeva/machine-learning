
from inception.model import InceptionV3TL
from keras.applications.inception_v3 import preprocess_input
from utils import preprocessing
from utils import train

TRAIN_DIR = 'C:\\Users\\helga_sh\\PycharmProjects\\asl-alphabet\\machine-learning\\chars74k\\EnglishImg\\Train'
VAL_DIR = 'C:\\Users\\helga_sh\\PycharmProjects\\asl-alphabet\\machine-learning\\chars74k\\EnglishImg\\Val'
WIDTH = 299
HEIGHT = 299
BATCH_SIZE = 32

model = InceptionV3TL()

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
