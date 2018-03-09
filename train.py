import argparse
import os
import shutil
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.optimizers import Adam
from functools import partial
import data
import degradations as degrad
import nets
from PIL import Image

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.48
set_session(tf.Session(config=config))

parser = argparse.ArgumentParser()
parser.add_argument("--VERSION", help="name for checkpoints and logs directory",  type=str, default='test')
parser.add_argument("--ORIGINAL_DIR", help="directory path with original images",  type=str, default='test')
args = parser.parse_args()


n_workers = 16
CHECKPOINT = 'checkpoint/%s' % args.VERSION
LOGDIR = 'logs/%s' % args.VERSION
if os.path.exists(CHECKPOINT): shutil.rmtree(CHECKPOINT)
if os.path.exists(LOGDIR): shutil.rmtree(LOGDIR)
os.makedirs(CHECKPOINT, exist_ok=True)


input_shape = (None, None, 3)
batch_size = 64
patch_size = 64
initial_lr = 0.001
n_epochs = 50


# blurring
kernels = []
for kern_size in [3, 5, 7, 9]:
    kernels.append(degrad.defocus_kernel(kern_size))
    for kern_direction in [0, 1, 2, 3]:
        kernels.append(degrad.motion_kernel(kern_size, kern_direction))
degrade_funs = [partial(degrad.blur, kernel=i) for i in kernels]

paths = np.array([os.path.join(args.ORIGINAL_DIR, i) for i in os.listdir(args.ORIGINAL_DIR)])
train_paths =[i for i in paths if "_orig" in i]


# JPEG
# degrade_funs = [partial(degrad.compress_jpeg, quality=i) for i in range(10,100,20)]
# train_paths = np.array([os.path.join(train_images_dir, i) for i in os.listdir(train_images_dir)])


# SUPER RES
# degrade_funs = [degrad.bicubic_restoration]
# train_paths = np.array([os.path.join(train_images_dir, i) for i in os.listdir(train_images_dir)])


train_images = data.mp_handler(Image.open, train_paths, n_workers)
train_batches_epoch = len(train_paths) // batch_size

train_generator = data.data_generator(train_images, degrade_funs, patch_size=patch_size, batch_size=batch_size)


model = nets.network(input_shape, K=12)
model.compile(optimizer=Adam(initial_lr), loss="mean_squared_error")

checkpointer = ModelCheckpoint(filepath='%s/check_{epoch:02d}.ckpt' % CHECKPOINT, verbose=1, period=2)
lrate = LearningRateScheduler(lambda epoch: initial_lr * 0.9 ** epoch)
tensorboard = TensorBoard(log_dir=LOGDIR, histogram_freq=0,
                          write_graph=True, write_grads=True,
                          batch_size=batch_size)

model.fit_generator(train_generator, steps_per_epoch=train_batches_epoch, epochs=n_epochs,
                    max_queue_size=128, callbacks=[checkpointer, lrate, tensorboard],
                    use_multiprocessing=False, workers=n_workers)

