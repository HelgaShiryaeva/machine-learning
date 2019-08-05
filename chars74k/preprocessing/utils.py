from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt
from random import shuffle


def load_data(folder_path, imsize):
    data = []
    for class_folder in tqdm(os.listdir(folder_path)):
        class_folder_path = os.path.join(folder_path, class_folder)
        label = int(class_folder.split("Sample")[1])
        for img in tqdm(os.listdir(class_folder_path)):
            image_path = os.path.join(class_folder_path, img)
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (imsize, imsize))
            data.append([img, label])
    print('Loaded successfully')
    return data


def plot_random_samples(labeled_images, row_size, col_size):
    fig = plt.figure()

    images_copy = list(labeled_images)
    shuffle(images_copy)
    images_copy = images_copy[: row_size * col_size]

    for num, data in enumerate(images_copy):
        img_data = data[0]
        label = data[1]
        y = fig.add_subplot(row_size, col_size, num + 1)
        y.imshow(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
        plt.title(label)
        y.get_xaxis().set_visible(False)
        y.get_yaxis().set_visible(False)
    plt.show()