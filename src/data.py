import configparser
import os

from keras.utils import image_utils
import matplotlib.pyplot as plt


config = configparser.ConfigParser()
config.read('config.ini')
IMG_SIZE = (int(config['IMG']['width']), int(config['IMG']['height']))
IMG_PATH = config['IMG']['path']

def show_image(img_path):
    """
    show a image from img_path
    """
    img = image_utils.load_img(img_path, target_size=IMG_SIZE)
    plt.imshow(img)
    plt.show()

def show_sample_images():
    files = os.listdir(IMG_PATH)
    np.random.shuffle(files)
    n = 8
    plt.figure(figsize=(10, 10))
    for i in range(n*n):
        img = image_utils.load_img('../data/UTKFace/'+files[i])
        plt.subplot(n, n, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.show()

show_sample_images()