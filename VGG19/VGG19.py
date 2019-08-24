import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications.vgg19 import VGG19
from keras import backend as K
from keras.utils import np_utils

img_width, img_height = 224, 224
checkpoint_filepath = "VGG19_weights.hdf5"
train_data_dir = 'C:/Users/Gebruiker/Machine Learning/Group/train'
validation_data_dir = 'C:/Users/Gebruiker/Machine Learning/Group/validation'
test_data_dir = 'C:/Users/Gebruiker/Machine Learning/Group/test'
nb_train_samples = 2656
nb_validation_samples = 896
nb_test_samples = 3550
epochs = 25
batch_size = 16

if K.image_data_format() == 'channels_first':
	input_shape = (3, 224, 224)
else:
	input_shape = (224, 224, 3)


def save_bottlebeck_features():
	datagen = ImageDataGenerator(rescale=1. / 255)

	# build the VGG19 network
	model = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)

	generator = datagen.flow_from_directory(
		train_data_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode=None,
		shuffle=False)
	bottleneck_features_train = model.predict_generator(generator, nb_train_samples // batch_size)
	np.save(open('VGG19_features_train.npy', 'wb'), bottleneck_features_train)

	generator = datagen.flow_from_directory(
		validation_data_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode=None,
		shuffle=False)
	bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples // batch_size)
	np.save(open('VGG19_features_validation.npy', 'wb'), bottleneck_features_validation)


def train_top_model():
	train_data = np.load(open('VGG19_features_train.npy', 'rb'))
	train_labels = np.array([0] * 235 +
	                        [1] * 147 +
	                        [2] * 341 +
	                        [3] * 60 +
	                        [4] * 386 +
	                        [5] * 270 +
	                        [6] * 381 +
	                        [7] * 280 +
	                        [8] * 187 +
	                        [9] * 328 +
	                        [10] * 41)

	validation_data = np.load(open('VGG19_features_validation.npy', 'rb'))
	validation_labels = np.array([0] * 79 +
	                             [1] * 50 +
	                             [2] * 116 +
	                             [3] * 21 +
	                             [4] * 129 +
	                             [5] * 90 +
	                             [6] * 129 +
	                             [7] * 94 +
	                             [8] * 63 +
	                             [9] * 111 +
	                             [10] * 14)
	train_labels = np_utils.to_categorical(train_labels)
	validation_labels = np_utils.to_categorical(validation_labels)

	model = Sequential()
	model.add(Flatten(input_shape=train_data.shape[1:]))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(11, activation='softmax'))

	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

	model.summary()

	checkpointer = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

	model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(validation_data, validation_labels), callbacks=[checkpointer])


#save_bottlebeck_features()
train_top_model()
