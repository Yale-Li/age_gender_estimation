import sys
import configparser

import numpy as np
from keras.utils import image_utils
from keras import models
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config.read('config.ini')
img_size = (int(config['IMG']['width']), int(config['IMG']['height']))

def img_to_tensor(img_path):
  img = image_utils.load_img(img_path, target_size=img_size)
  img_tensor = image_utils.img_to_array(img)
  img_tensor = np.expand_dims(img_tensor, axis=0)
  img_tensor /= 255
  print(img_tensor.shape)
  return img_tensor


def show_intermediate_model(img_path, model_path, level=0, channel=1):
  model = models.load_model(model_path)
  layer_outputs = [layer.output for layer in model.layers]
  new_model = models.Model(inputs=model.input, outputs=layer_outputs)
  activations = new_model.predict(img_to_tensor(img_path))
  layer = activations[level]
  print(layer.shape)

  plt.matshow(layer[0, :, :, channel], cmap='viridis')
  plt.show()

def show_image(img_path):
  img = image_utils.load_img(img_path, target_size=img_size)
  plt.imshow(img)
  plt.show()


if __name__ == '__main__':
  if len(sys.argv) > 1:
    show_image(sys.argv[1])
    show_intermediate_model(sys.argv[1], 'dogs_and_cats.h5')
  