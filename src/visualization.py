import sys
import configparser

import tensorflow as tf
from keras.utils import image_utils
from keras import models, Model
import matplotlib.pyplot as plt
import numpy as np


config = configparser.ConfigParser()
config.read('config.ini')
IMG_SIZE = (int(config['IMG']['width']), int(config['IMG']['height']))
IMG_PATH = config['IMG']['path']


def img_to_tensor(img_path):
    """
    load a image from img_path, turn into Numpy Array
    """
    img = image_utils.load_img(img_path, target_size=IMG_SIZE)
    img_tensor = image_utils.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    # print(img_tensor.shape)
    return img_tensor


def show_intermediate_model(
    model,
    img_path, 
    level=1
):
    layer_outputs = [layer.output for layer in model.layers[:]]
    new_model = models.Model(inputs=model.input, outputs=layer_outputs)
    feature_extractor = new_model.predict(img_to_tensor(img_path))
    layer = feature_extractor[level]

    layer_depth = layer.shape[-1]
    n = int(np.ceil(np.sqrt(layer_depth)))
    plt.figure(figsize=(10,10))
    
    for i in range(layer_depth):
        ax = plt.subplot(n, n, i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(layer[0, :, :, i])
    plt.suptitle(f'{level}')
    plt.show()
    return


def show_filters(model, layer_name, filter_index):
    """
    https://keras.io/examples/vision/visualizing_what_convnets_learn/
    """
    layer = model.get_layer(name=layer_name)
    feature_extractor = Model(inputs=model.inputs, outputs=layer.output)

    def compute_loss(input_image, filter_index):
        activation = feature_extractor(input_image)
        # We avoid border artifacts by only involving non-border pixels in the loss.
        filter_activation = activation[:, 2:-2, 2:-2, filter_index]
        return tf.reduce_mean(filter_activation)

    # @tf.function
    def gradient_ascent_step(img, filter_index, learning_rate):
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = compute_loss(img, filter_index)
        # Compute gradients.
        grads = tape.gradient(loss, img)
        # Normalize gradients.
        grads = tf.math.l2_normalize(grads)
        img += learning_rate * grads
        return loss, img

    def initialize_image():
        # We start from a gray image with some random noise
        img = tf.random.uniform((1, IMG_SIZE[0], IMG_SIZE[1], 3))
        # ResNet50V2 expects inputs in the range [-1, +1].
        # Here we scale our random inputs to [-0.125, +0.125]
        return (img - 0.5) * 0.25

    def deprocess_image(img):
        # Normalize array: center on 0., ensure variance is 0.15
        img -= img.mean()
        img /= img.std() + 1e-5
        img *= 0.15

        # Center crop
        img = img[25:-25, 25:-25, :]

        # Clip to [0, 1]
        img += 0.5
        img = np.clip(img, 0, 1)

        # Convert to RGB array
        img *= 255
        img = np.clip(img, 0, 255).astype("uint8")
        return img

    def visualize_filter(filter_index = 0):
        # We run gradient ascent for 20 steps
        iterations = 30
        learning_rate = 10.0
        img = initialize_image()
        for iteration in range(iterations):
            loss, img = gradient_ascent_step(img, filter_index, learning_rate)

        # Decode the resulting input image
        img = deprocess_image(img[0].numpy())
        return loss, img
    # return visualize_filter(filter_index)[1]
    # Compute image inputs that maximize per-filter activations
    # for the first 64 filters of our target layer
    all_imgs = []
    for filter_index in range(6):
        print("Processing filter %d" % (filter_index,))
        loss, img = visualize_filter(filter_index)
        all_imgs.append(img)

    # Build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    n = 4
    cropped_width = IMG_SIZE[0] - 2 * 2
    cropped_height = IMG_SIZE[1] - 2 * 2
    width = n * cropped_width + (n - 1) * margin
    height = n * cropped_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # Fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img = all_imgs[i * n + j]
            print(img.shape)
            stitched_filters[
                (cropped_width + margin) * i : (cropped_width + margin) * i + cropped_width,
                (cropped_height + margin) * j : (cropped_height + margin) * j
                + cropped_height,
                :,
            ] = img
    plt.imshow(stitched_filters)
    plt.show()


def show_image(img_path):
    """
    show a image from img_path
    """
    img = image_utils.load_img(img_path, target_size=IMG_SIZE)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    model = models.load_model('../data/gender_model')
    layer_name = model.layers[2].name
    # print([layer.name for layer in model.layers])
    # show_filters(model, layer_name, 0)

    if len(sys.argv) > 1:
        # show_image(sys.argv[1])
        for i in range(41):
            show_intermediate_model(model, sys.argv[1], i)
