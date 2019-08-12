from data_generator import DataGenerator
from model import InceptionV3TL

data_gen = DataGenerator()
model = InceptionV3TL()
train_generator, validation_generator = data_gen.test_train_generators()
x_batch, y_batch = next(train_generator)
DataGenerator.plot_random_samples(x_batch, y_batch)
history = model.train(train_generator, validation_generator)
model.plot_training(history)
