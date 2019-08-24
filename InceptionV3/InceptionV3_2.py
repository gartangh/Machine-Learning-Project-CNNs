import numpy as np
from time import localtime, strftime
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K


img_width, img_height = 224, 224
time = localtime()
#checkpoint_filepath = strftime('InceptionV3/Checkpoints/InceptionV3_2_weights_%Y%m%d_%H%M%S.hdf5', time)
checkpoint_filepath = "InceptionV3/Checkpoints/InceptionV3_2_weights_20181115_101301.hdf5"
checkpoint_filepath_fine_tuned = strftime(
	'InceptionV3/Checkpoints/InceptionV3_2_Fine_Tuned_weights_%Y%m%d_%H%M%S.hdf5', time)
log_dir = strftime('InceptionV3/Logs/InceptionV3_2_%Y%m%d_%H%M%S', time)
log_dir_fine_tuned = strftime('InceptionV3/Logs/InceptionV3_2_Fine_Tuned_%Y%m%d_%H%M%S', time)
train_data_dir = 'C:/Users/Gebruiker/Machine Learning/Group/train'
validation_data_dir = 'C:/Users/Gebruiker/Machine Learning/Group/validation'
test_data_dir = 'C:/Users/Gebruiker/Machine Learning/Group/test'
nb_train_samples = 2656
nb_validation_samples = 896
nb_test_samples = 3550
epochs = 50
batch_size = 16

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 11 classes
predictions = Dense(11, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
	layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

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

model.summary()

checkpointer = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
tensorboard = TensorBoard(log_dir=log_dir)

# train the model on the new data for a few epochs
#model.fit_generator(
#	train_generator,
#	steps_per_epoch=nb_train_samples // batch_size,
#	epochs=epochs,
#	validation_data=validation_generator,
#	validation_steps=nb_validation_samples // batch_size,
#	callbacks=[checkpointer, tensorboard])

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
	print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
	layer.trainable = False
for layer in model.layers[249:]:
	layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD

model.load_weights(checkpoint_filepath)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

checkpointer = ModelCheckpoint(checkpoint_filepath_fine_tuned, monitor='val_loss', verbose=1, save_best_only=True,
                               mode='auto')
tensorboard = TensorBoard(log_dir=log_dir_fine_tuned)

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(
	train_generator,
	steps_per_epoch=nb_train_samples // batch_size,
	epochs=epochs,
	validation_data=validation_generator,
	validation_steps=nb_validation_samples // batch_size,
	callbacks=[checkpointer, tensorboard])

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
	test_data_dir,
	batch_size=1,
	target_size=(img_width, img_height),
	class_mode=None,
	shuffle=False)

features_test = model.predict_generator(test_generator, nb_test_samples)
print(features_test.shape)
np.save(open(strftime('Inceptionv3/InceptionV3_3_features_test_%Y%m%d_%H%M%S', time), 'wb'), features_test)
