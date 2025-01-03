# importing libraries
from glob import glob
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dropout, Dense
from keras.applications import ResNet152V2
from keras.applications.resnet_v2 import preprocess_input
from keras.optimizers import Adam
import tensorflow as tf
import kagglehub

# setup 
train_dir  = "../dataset/Plant_leave_diseases_dataset_with_augmentation"

print("Number of Images are: {}".format(len(glob(train_dir + '/*/*'))))

# data augmentation
aug = ImageDataGenerator(preprocessing_function = preprocess_input,
	validation_split = 0.20,
	rescale = 1./255)

batch_size = 32
training_set = aug.flow_from_directory(train_dir,
	target_size = (224, 224),
	batch_size = batch_size,
	class_mode = "categorical",
	subset = "training")
test_set = aug.flow_from_directory(train_dir,
	target_size = (224, 224),
	batch_size = batch_size,
	class_mode = "categorical",
	subset = "validation")

# define architecture
baseModel = ResNet152V2(weights = "imagenet", include_top = False, input_shape = (224, 224, 3))
headModel = baseModel.output
headModel = GlobalAveragePooling2D()(headModel)
headModel = Dropout(0.25)(headModel)
headModel = Dense(39, activation='sigmoid', name = "resnet152v2_dense")(headModel)

model = Model(inputs = baseModel.input, outputs = headModel, name = "ResNet152V2")

model.trainable = True
print(model.summary())

# define criteria for stopping. we will stop training if validation accuracy got reached 98%
class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('val_accuracy') > 0.98):
			print("\nReached 98% accuracy so cancelling training!")
			self.model.stop_training = True

callbacks = myCallback()

# compile model
model.compile(optimizer = Adam(learning_rate = 0.005), loss = 'categorical_crossentropy', 
	metrics=['accuracy'])

# start training
H = model.fit_generator(training_set,
	steps_per_epoch = training_set.samples//batch_size,
	validation_data = test_set,
	epochs = 10,
	validation_steps = test_set.samples//batch_size,
	callbacks = [callbacks],
	verbose = 1)

# save the model to file
model.save('../models/resnet152v2.h5')