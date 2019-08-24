import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras import backend as K

img_width, img_height = 224, 224
checkpoint_filepath = "VGG16/VGG16_2_weights.hdf5"
train_data_dir = 'C:/Users/Gebruiker/Machine Learning/Group/train'
validation_data_dir = 'C:/Users/Gebruiker/Machine Learning/Group/validation'
test_data_dir = 'C:/Users/Gebruiker/Machine Learning/Group/test'
nb_train_samples = 2656
nb_validation_samples = 896
nb_test_samples = 3550
epochs = 25
batch_size = 16

if K.image_data_format() == 'channels_first':
	input_shape = (3, img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)

# build the VGG16 network
model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(11, activation='softmax'))

new_model = Sequential()
for l in model.layers:
	new_model.add(l)

# add the model on top of the convolutional base
new_model.add(top_model)

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in new_model.layers[:18]:
	layer.trainable = False

new_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
	rotation_range=40,
	width_shift_range=0.2,
	height_shift_range=0.2,
	rescale=1. / 255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest')

# this is the augmentation configuration we will use for testing: only rescaling
validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
	train_data_dir,
	batch_size=batch_size,
	target_size=(img_width, img_height),
	class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
	validation_data_dir,
	batch_size=batch_size,
	target_size=(img_width, img_height),
	class_mode='categorical')

new_model.summary()

checkpointer = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

new_model.fit_generator(
	train_generator,
	steps_per_epoch=nb_train_samples // batch_size,
	epochs=epochs,
	validation_data=validation_generator,
	validation_steps=nb_validation_samples // batch_size,
	callbacks=[checkpointer])
