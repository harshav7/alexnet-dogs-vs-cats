import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from config import dogs_vs_cats_config as config
from utilities.preprocessing import ImageToArrayPreprocessor
from utilities.preprocessing import SimplePreprocessor
from utilities.preprocessing import PatchPreprocessor
from utilities.preprocessing import MeanPreprocessor
from utilities.callbacks import TrainingMonitor
from utilities.io import HDF5DatasetGenerator
from utilities.nn.cnn import AlexNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import json
import os
from imutils import paths
from keras.backend.tensorflow_backend import set_session
import shutil
import tensorflow as tf

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices()) # list of DeviceAttributes
# configg = tensorflow.ConfigProto()
# configg.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# configg.log_device_placement = True  # to log device placement (on which device the operation ran)
# sess = tensorflow.Session(config=configg)
# set_session(sess)  # set this TensorFlow session as the default session for Keras

#for tensorflow 2
import tensorflow as tf
configg = tf.compat.v1.ConfigProto() 
configg.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=configg)

# aug = ImageDataGenerator(rotation_range=20,zoom_range=0.1,
#     width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
#     horizontal_flip = True, fill_mode = "nearest")
# IMAGES_PATH = "../datasets/kaggle_dogs_vs_cats/test1"
# catdir ="../datasets/kaggle_dogs_vs_cats/cat"
# trainPaths = list(paths.list_images(IMAGES_PATH))
# trainLabels = [p.split(os.path.sep)[-1].split(".")[0] for p in trainPaths]
# for p in trainPaths:
#     l = p.split(os.path.sep)[-1].split(".")[0]
#     if l =="cat":
#         print("cat")
#         shutil.move(p,catdir)



# image_gen_train = ImageDataGenerator(
#                     validation_split = 0.2,
#                     rescale=1./255,
#                     rotation_range=45,
#                     width_shift_range=.15,
#                     height_shift_range=.15,
#                     horizontal_flip=True,
#                     zoom_range=0.1
#                     )
batch_size = 1
img_height =227
img_width =227
train_data_dir = "../datasets/kaggle_dogs_vs_cats/train"


train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    # class_mode='binary',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    # class_mode='binary',
    subset='validation') # set as validation data

# model.fit_generator(
#     train_generator,
#     steps_per_epoch = train_generator.samples // batch_size,
#     validation_data = validation_generator, 
#     validation_steps = validation_generator.samples // batch_size,
#     epochs = nb_epochs)


# initialize the optimizer
print("[INFO] compiling model...")
opt = Adam(lr=1e-3)
model = AlexNet.build(width=227, height=227, depth=3,
    classes=2, reg=0.0002)
model.compile(loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# construct the set of callbacks
path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(
    os.getpid())])

callbacks = [TrainingMonitor(path)]

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=validation_generator, validation_steps=validation_generator.samples // batch_size
    )

print("acc = ",history.history['accuracy'])
print("val_acc = ",history.history['val_accuracy'])

print("loss = ",history.history['loss'])
print("val_loss =" ,history.history['val_loss'])

# save the model to file
print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

