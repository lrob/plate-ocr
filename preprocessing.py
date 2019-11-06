from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

import numpy as np
import matplotlib.pyplot as plt

import pathlib

LETTERS = np.array(['A','B','C','D','E','F','G','H','L','M','P','R','S','T','V','Z','W','Y','X','K','J','U','Q','I'])
NUMBERS = np.array(['1','2','3','4','5','6','7','8','9','0'])


BATCH_SIZE = 32
IMG_HEIGHT = 110
IMG_WIDTH = 520
#STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

data_dir = "/home/roberto/Documents/2019-11-04-cnn-ocr/plates"
data_dir = pathlib.Path(data_dir)

def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, '/')
    # The second to last is the class-directory
    filename = parts[-1]
    character = tf.strings.substr(filename, 0, 1) 
    labelPart = character == LETTERS
    label = labelPart

    character = tf.strings.substr(filename, 1, 1)
    labelPart = character == LETTERS
    label = tf.concat([label, labelPart], 0)

    character = tf.strings.substr(filename, 2, 1)
    labelPart = character == NUMBERS
    label = tf.concat([label, labelPart], 0)

    character = tf.strings.substr(filename, 3, 1)
    labelPart = character == NUMBERS
    label = tf.concat([label, labelPart], 0)

    character = tf.strings.substr(filename, 4, 1)
    labelPart = character == NUMBERS
    label = tf.concat([label, labelPart], 0)

    character = tf.strings.substr(filename, 5, 1)
    labelPart = character == LETTERS
    label = tf.concat([label, labelPart], 0)

    character = tf.strings.substr(filename, 6, 1)
    labelPart = character == LETTERS
    label = tf.concat([label, labelPart], 0)

    return label

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label


list_ds = tf.data.Dataset.list_files(str(data_dir/'*.jpg'))

# for f in list_ds.take(50):
#    print(f.numpy())
#    get_label(f)

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
#tf.enable_eager_execution()
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

 
  

# image_count = len(list(data_dir.glob('*.jpg')))

# image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)





