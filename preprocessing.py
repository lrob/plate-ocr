from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

import numpy as np
import matplotlib.pyplot as plt
import os

import pathlib

LETTERS = np.array(['A','B','C','D','E','F','G','H','L','M','P','R','S','T','V','Z','W','Y','X','K','J','U','Q','I'])
NUMBERS = np.array(['1','2','3','4','5','6','7','8','9','0'])



BATCH_SIZE = 32
REAL_IMG_HEIGHT = 110
REAL_IMG_WIDTH = 520
img_height = 96
img_width = int(REAL_IMG_WIDTH * img_height / REAL_IMG_HEIGHT)
#STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

data_dir = "plates/"
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
  return tf.image.resize(img, [img_width, img_height])

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

DATASET_SIZE = 99998
TRAIN_PERC_SIZE = 0.8

train_size = int(DATASET_SIZE * TRAIN_PERC_SIZE)
train_ds = labeled_ds.take(train_size)
dev_ds = labeled_ds.skip(train_size) 

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds

train_batches = prepare_for_training(train_ds, "./train_batches_caches.ch", SHUFFLE_BUFFER_SIZE)
dev_batches = dev_ds.batch(BATCH_SIZE)

for image_batch, label_batch in train_batches.take(1):
   pass

print("image batch shape:", image_batch.shape)

IMG_SHAPE = (img_width, img_height, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')


feature_batch = base_model(image_batch)
print("feature map batch shape:", feature_batch.shape)

#base_model.trainable = False

# Let's take a look to see how many layers are in the base model
#print("Number of layers in the base model: ", len(base_model.layers))

# Fine tune from this layer onwards
#fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
#for layer in base_model.layers[:fine_tune_at]:
#  layer.trainable =  False



base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print("Dimensio after avarate pooling:", feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(LETTERS.shape[0] * 4 + NUMBERS.shape[0] * 3, activation="sigmoid")
prediction_batch = prediction_layer(feature_batch_average)
print("Dimension after dense layer:", prediction_batch.shape)

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

def evaluateCharacter(y_true, y_pred, begin_character, end_character):
  true_class = tf.keras.backend.argmax(y_true[:,begin_character:end_character], axis=1)
  predicted_class = tf.keras.backend.argmax(y_pred[:,begin_character:end_character], axis=1)
  return tf.keras.backend.cast(predicted_class == true_class, dtype = tf.uint8)

#evChar = evaluateCharacter([0,0,0,1], [0,0,0,1], 0, 4) + evaluateCharacter([0,0,0,1], [0,0,0,1], 0, 4)

#print("evChar", evChar)
#print("evaluate character.shape" , evChar.shape)


def getNumCorretCharacters(y_true, y_pred):
  current_index = 0
  num_correct_character_predicted = evaluateCharacter(y_true, y_pred, current_index, current_index + LETTERS.shape[0])
  current_index += LETTERS.shape[0]
  num_correct_character_predicted += evaluateCharacter(y_true, y_pred, current_index, current_index + LETTERS.shape[0]) 
  current_index += LETTERS.shape[0]
  num_correct_character_predicted += evaluateCharacter(y_true, y_pred, current_index, current_index + NUMBERS.shape[0]) 
  current_index += NUMBERS.shape[0]
  num_correct_character_predicted += evaluateCharacter(y_true, y_pred, current_index, current_index + NUMBERS.shape[0])
  current_index += NUMBERS.shape[0]
  num_correct_character_predicted += evaluateCharacter(y_true, y_pred, current_index, current_index + NUMBERS.shape[0])
  current_index += NUMBERS.shape[0]
  num_correct_character_predicted += evaluateCharacter(y_true, y_pred, current_index, current_index + LETTERS.shape[0])
  current_index += LETTERS.shape[0]
  num_correct_character_predicted += evaluateCharacter(y_true, y_pred, current_index, current_index + LETTERS.shape[0]) 
  return num_correct_character_predicted

y_true_arg = np.zeros((32,126))
y_pred_arg = np.zeros((32,126))

y_true_arg[:,9] = y_true_arg[:,29] = y_true_arg[:,50] = y_true_arg[:,60] = y_true_arg[:,70] = y_true_arg[:,80] = y_true_arg[:,120] = 1 
y_pred_arg[:,9] = y_pred_arg[:,29] = y_pred_arg[:,50] = y_pred_arg[:,60] = y_pred_arg[:,70] = y_pred_arg[:,80] = y_pred_arg[:,120] = 1 
#print("getNumCorrectCharacters", getNumCorretCharacters(y_true_arg, y_pred_arg))

y_true_arg = tf.convert_to_tensor(y_true_arg)
y_pred_arg = tf.convert_to_tensor(y_pred_arg)


NUM_CHARACTERS_PLATE = 7
def characterAccuracy(y_true, y_pred):
  num_correct_character_predicted = getNumCorretCharacters(y_true, y_pred)
  return num_correct_character_predicted/NUM_CHARACTERS_PLATE
  #return tf.keras.backend.mean(y_true)

print("characterAccuracy", characterAccuracy(y_true_arg, y_pred_arg))

def plateAccuracy(y_true, y_pred):
  num_correct_character_predicted = getNumCorretCharacters(y_true, y_pred) 
  return tf.keras.backend.cast(tf.math.equal(num_correct_character_predicted, tf.constant(NUM_CHARACTERS_PLATE, dtype=tf.uint8)), dtype = tf.uint8)

print("plateAccuracy", plateAccuracy(y_true_arg, y_pred_arg))


model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.99, beta_2=0.9999, epsilon=None, decay=0.0, amsgrad=False),
              #metrics=[characterAccuracy])
              metrics=[plateAccuracy, characterAccuracy])

model.summary()

print("Number or variable to be trained:", len(model.trainable_variables))

CHECKPOINT_PATH = "checkpoint/cp.ckpt"
checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=2000)


if(os.path.isdir(checkpoint_dir) and os.listdir(checkpoint_dir) != []):
  print("Loading weights")
  model.load_weights(CHECKPOINT_PATH)
else:
  print("Starting training from scratch")

initial_epochs = 100
steps_per_epoch = train_size//BATCH_SIZE
validation_steps = 3

print("steps per epoch: ", steps_per_epoch)

loss0,accuracy0_0, accuracy1_0 = model.evaluate(dev_batches, steps = validation_steps)

print("initial loss, plate accuracy and character accuracy: ", loss0, accuracy0_0, accuracy1_0)

history = model.fit(train_batches,
                    epochs=initial_epochs,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=dev_batches,
                    callbacks=[cp_callback])

MODEL_PATH = checkpoint_dir + "/model.h5"
model.save(MODEL_PATH) 








