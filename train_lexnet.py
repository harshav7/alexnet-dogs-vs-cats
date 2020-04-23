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

from keras.backend.tensorflow_backend import set_session

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

aug = ImageDataGenerator(rotation_range=20,zoom_range=0.1,
    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
    horizontal_flip = True, fill_mode = "nearest")

means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

batch_size = 1
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, batch_size, aug=aug,
    preprocessors=[pp, mp, iap], classes=2)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, batch_size ,
    preprocessors=[sp, mp, iap], classes=2)

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

# train the network
# model.fit_generator(
#     trainGen.generator(),
#     steps_per_epoch=trainGen.numImages // (128*4),
#     validation_data=valGen.generator(),
#     validation_steps=valGen.numImages // (128*4),
#     epochs=75,
#     max_queue_size=128 , workers = 0,
#     callbacks=callbacks, verbose=1)

model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // (batch_size),
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // (batch_size),
    epochs=1,
    max_queue_size=batch_size ,
    callbacks=callbacks, verbose=1)

# save the model to file
print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)
# close the HDF5 datasets
trainGen.close()
valGen.close()
