from io import BytesIO
import numpy as np
from PIL import Image, ImageFilter
from scipy.signal import convolve2d
from skimage.draw import circle


def compress_jpeg(img, quality):
    #img - is PIL img
    faux_file = BytesIO()
    img.save(faux_file, format='jpeg', subsampling=0, quality=quality)
    return Image.open(BytesIO(faux_file.getvalue()))


def bicubic_downsample(img, r=4):
    input = img.resize((img.width // r, img.height // r), resample=Image.BICUBIC)
    return input


def bicubic_restoration(img, r_bicubic=4):
    assert (img.width // r_bicubic * r_bicubic == img.width) & (img.height // r_bicubic * r_bicubic == img.height)
    small_img = img.resize((img.width // r_bicubic, img.height // r_bicubic), resample=Image.BICUBIC)
    restored = small_img.resize((small_img.width * r_bicubic, small_img.width * r_bicubic), resample=Image.BICUBIC)
    return restored


# regular convolutions in pil support only kernels [3,3] or [5,5]
def blur_pil(img, sigma = 20):
    return img.fiter(ImageFilter.Kernel((3, 3), np.random.normal(loc=0.0, scale=sigma / 255, size=9)))


def blur(img, kernel):
    imgarray = np.array(img, dtype="float32")
    convolved = np.zeros(shape = np.shape(imgarray), dtype="uint8")
    for i in range(np.shape(convolved)[-1]):
        convolved[:,:,i] = convolve2d(imgarray[:,:,i], kernel, mode='same', fillvalue=255.0).astype("uint8")
    img = Image.fromarray(convolved)
    return img


def motion_kernel(dim, direction):
    # direction 0 = horizontal, 1 = vertical, 2 = diagonal, 3 = back_diagonal
    assert dim%2 != 0
    kernel = np.zeros((dim, dim), dtype=np.float32)
    if direction == 0:
        kernel[dim // 2, :] = 1
    elif direction == 1:
        kernel[:, dim // 2] = 1
    elif direction == 2:
        kernel[range(dim), (np.arange(dim) + 1) * -1] = 1
    elif direction == 3:
        kernel[range(dim), range(dim)] = 1
    else: AssertionError("wrong direction specified")

    return normalize_kernel(kernel)


def defocus_kernel(dim):
    assert dim % 2 != 0
    kernel = np.zeros((dim, dim), dtype=np.float32)
    circleCenterCoord = dim // 2
    circleRadius = circleCenterCoord +1 #if dim % 2 else circleCenterCoord

    rr, cc = circle(circleCenterCoord, circleCenterCoord, circleRadius)
    kernel[rr, cc] = 1

    if (dim == 3) or (dim == 5):
        kernel = Adjust(kernel, dim)

    return normalize_kernel(kernel)


def normalize_kernel(kernel):
    if np.sum(kernel) > 1.0001:
        normalizationFactor = np.count_nonzero(kernel)
        kernel = kernel / normalizationFactor
    return kernel


def Adjust(kernel, kernelwidth):
    kernel[0,0] = 0
    kernel[0,kernelwidth-1]=0
    kernel[kernelwidth-1,0]=0
    kernel[kernelwidth-1, kernelwidth-1] =0
    return kernel

