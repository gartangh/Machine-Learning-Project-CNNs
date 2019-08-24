import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

img_width, img_height = 224, 224
checkpoint_filepath = "InceptionV3_2_Fine_Tuned_weights.hdf5"
test_data_dir = 'C:/Users/Gebruiker/Machine Learning/Group/test'
nb_test_samples = 3550

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

model.load_weights('InceptionV3_2_Fine_Tuned_weights.hdf5')

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
	test_data_dir,
	batch_size=1,
	target_size=(img_width, img_height),
	class_mode=None,
	shuffle=False)

features_test = model.predict_generator(test_generator, nb_test_samples)
print(features_test.shape)
np.save(open('InceptionV3_3_features_test', 'wb'), features_test)
