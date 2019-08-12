from tqdm import tqdm
from shutil import copyfile
import os
import cv2
import matplotlib.pyplot as plt
from random import shuffle

CONST_TRAIN = 'Train'
CONST_VAL = 'Val'
CONST_TEST = 'Test'
CONST_ALL = 'All'


def load_data(folder_path, im_size):
    data = []
    for class_folder in tqdm(os.listdir(folder_path)):
        class_folder_path = os.path.join(folder_path, class_folder)
        label = int(class_folder.split("Sample")[1])
        for img in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, img)
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (im_size, im_size))
            data.append([img, label])
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


def copy_imgs(src, dst, img_names):
    for img_name in img_names:
        src_path = os.path.join(src, img_name)
        dst_path = os.path.join(dst, img_name)
        copyfile(src_path, dst_path)


def preprocess_images(folder_path):
    class_folders = os.listdir(os.path.join(folder_path, CONST_ALL))
    type_folders = [CONST_TEST, CONST_TRAIN, CONST_VAL]

    for type_folder in type_folders:
        if os.path.isdir(os.path.join(folder_path, type_folder)):
            print('Can not process.\n' + '"' + type_folder + '" already exist.')
            return

        for class_folder in class_folders:
            os.makedirs(os.path.join(folder_path, type_folder, class_folder))

    for class_folder in tqdm(class_folders):
        imgs = os.listdir(os.path.join(folder_path, CONST_ALL, class_folder))
        shuffle(imgs)

        train_imgs = imgs[0: int(len(imgs) * 0.7)]
        val_imgs = imgs[int(len(imgs) * 0.7): int(len(imgs) * 0.8)]
        test_imgs = imgs[int(len(imgs) * 0.8): len(imgs)]

        src_path = os.path.join(folder_path, CONST_ALL, class_folder)

        copy_imgs(src_path, os.path.join(folder_path, CONST_TRAIN, class_folder), train_imgs)
        copy_imgs(src_path, os.path.join(folder_path, CONST_VAL, class_folder), val_imgs)
        copy_imgs(src_path, os.path.join(folder_path, CONST_TEST, class_folder), test_imgs)
