from keras.models import model_from_json
import matplotlib.pyplot as plt


def save_model(model, json_filename, weights_filename):
    model_json = model.to_json()
    with open(json_filename, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(weights_filename)
    print("Saved model to disk")


def load_model(json_filename, weights_filename):
    json_file = open(json_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_filename)
    print("Loaded model from disk")

    return model


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