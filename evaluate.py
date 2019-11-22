import tensorflow as tf
import os
import pathlib
import numpy as np

TEST_DS_PATH = "./plates-test"
test_ds_path = pathlib.Path(TEST_DS_PATH)

def plateAccuracy(y_true, y_pred):
  return 1

def characterAccuracy(y_true, y_pred):
  return 1

 
  
  
  num_correct_character_predicted = evaluateCharacter(y_true, y_pred, current_index, current_index + LETTERS.shape[0])
  current_index += LETTERS.shape[0]

LETTERS = np.array(['A','B','C','D','E','F','G','H','L','M','P','R','S','T','V','Z','W','Y','X','K','J','U','Q','I'])
NUMBERS = np.array(['1','2','3','4','5','6','7','8','9','0'])


REAL_IMG_HEIGHT = 110
REAL_IMG_WIDTH = 520
img_height = 96
img_width = int(REAL_IMG_WIDTH * img_height / REAL_IMG_HEIGHT)

def from_array_to_plate(prediction_array):
  plate = ''
  
  current_index = 0
  sub_array = prediction[0:LETTERS.shape[0]]
  character_idx = np.argmax(sub_array)
  plate += LETTERS[character_idx]
  current_index += LETTERS.shape[0]

  sub_array = prediction[0:LETTERS.shape[0]]
  character_idx = np.argmax(sub_array)
  plate +=  LETTERS[character_idx]
  current_index += LETTERS.shape[0]

  sub_array = prediction[0:NUMBERS.shape[0]]
  character_idx = np.argmax(sub_array)
  plate +=  NUMBERS[character_idx]
  current_index += NUMBERS.shape[0]

  sub_array = prediction[0:NUMBERS.shape[0]]
  character_idx = np.argmax(sub_array)
  plate +=  NUMBERS[character_idx]
  current_index += NUMBERS.shape[0]

  sub_array = prediction[0:NUMBERS.shape[0]]
  character_idx = np.argmax(sub_array)
  plate +=  NUMBERS[character_idx]
  current_index += NUMBERS.shape[0]

  sub_array = prediction[0:LETTERS.shape[0]]
  character_idx = np.argmax(sub_array)
  plate +=  LETTERS[character_idx]
  current_index += LETTERS.shape[0]

  sub_array = prediction[0:LETTERS.shape[0]]
  character_idx = np.argmax(sub_array)
  plate +=  LETTERS[character_idx]

  return plate

def decode_img(file_path):
  img = tf.io.read_file(file_path)
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [img_width, img_height])

list_ds = tf.data.Dataset.list_files(str(test_ds_path/'*.jpg'))
test_ds = list_ds.map(decode_img)
test_ds = test_ds.batch(1)

MODEL_PATH = "checkpoint/model.h5"
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"characterAccuracy": characterAccuracy, "plateAccuracy": plateAccuracy})

predictions = model.predict(test_ds)

for prediction in predictions:
  print(from_array_to_plate(prediction))