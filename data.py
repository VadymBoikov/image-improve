import numpy as np
import multiprocessing
import time


def mp_handler(fun, paths, workers = 4):
    p = multiprocessing.Pool(workers)
    ts = time.time()
    result = []
    for i, res in enumerate(p.imap(fun, paths)):
        result.append(res)
        if i % 1000 == 1:
            print("done %s in %s sec" % (i, time.time() - ts))
    return result


def random_crop(x, random_crop_size, sync_seed=None, **kwargs):
    # input is PIL Image
    # random_crop_size - integer of pixels
    w = x.width
    h = x.height
    rangew = (w - random_crop_size)
    rangeh = (h - random_crop_size)

    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)

    cropped = x.crop(box = (offsetw, offseth, offsetw+random_crop_size, offseth+random_crop_size))
    return cropped


def normalize_image(img_arr):
    return (img_arr / 255.) - 0.5


def denormalize_image(img_arr):
    return np.clip((img_arr + 0.5) * 255, 0, 255).astype(np.uint8)


def data_generator(images, degrade_fun, patch_size, batch_size=64):
    total_size = len(images)
    k = 0
    input_batch = np.zeros([batch_size, patch_size, patch_size, 3])
    output_batch = np.zeros([batch_size, patch_size, patch_size, 3])
    # if multiple degrade functions are passed
    _degrade_fun = np.random.choice(degrade_fun) if isinstance(degrade_fun, list) else degrade_fun

    while True:
        i = np.random.randint(total_size)
        output = random_crop(images[i], patch_size)
        input = _degrade_fun(output)

        input_arr = np.array(input)
        output_arr = np.array(output)
        input_arr = normalize_image(input_arr)
        output_arr = normalize_image(output_arr)

        input_batch[k, :, :, :] = input_arr
        output_batch[k, :, :, :] = output_arr
        k +=1
        if k == batch_size:
            k = 0
            if isinstance(degrade_fun, list):
                _degrade_fun = np.random.choice(degrade_fun)
            yield input_batch, output_batch
