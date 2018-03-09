import argparse
import os
import numpy as np
from keras.models import load_model
import data
from PIL import Image
from data import normalize_image, denormalize_image

parser = argparse.ArgumentParser()
parser.add_argument("--CHECKPOINT", help="path for checkpoint file",  type=str, default='test')
parser.add_argument("--CORRUPTED_DIR", help="directory path with corrupted images",  type=str, default='test')
parser.add_argument("--SAVE_DIR", help="directory path to save predictions",  type=str, default='test')
args = parser.parse_args()


os.makedirs(args.SAVE_DIR, exist_ok=True)
n_workers = 4

paths = np.array([os.path.join(args.CORRUPTED_DIR, i) for i in os.listdir(args.CORRUPTED_DIR)])
images = data.mp_handler(Image.open, paths, n_workers)

model = load_model(args.CHECKPOINT, compile=False)

print('starting predictions')
for i in range(len(images)):
    input_arr = [normalize_image(np.array(images[i]))]
    prediction = model.predict(input_arr)[0]

    Image.fromarray(denormalize_image(prediction)).save('%s/%s' % (args.SAVE_DIR, os.path.basename(paths[i])))
