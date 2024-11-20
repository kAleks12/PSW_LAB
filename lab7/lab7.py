import matplotlib.pyplot as plt
import numpy as np
import skimage.data
from pywt import wavedec
from skimage.transform import resize
from scipy.ndimage import gaussian_filter, correlate


def normalize(vector):
    new_vec = np.copy(vector)
    new_vec /= np.sum(new_vec)
    return new_vec


def zad1():
    signal = np.load('signal.npy')
    fig, ax = plt.subplots(8, 1, figsize=(10, 15))
    coefficients = wavedec(signal, 'db7', level=7)
    color = plt.cm.coolwarm(np.linspace(0, 1, 8))
    ax[0].plot(signal, color=color[0])
    ax[0].grid(True)
    ax[0].set_ylabel('signal')
    ax[0].spines['top'].set_visible(False)

    for i in range(1, 8):
        ax[i].plot(coefficients[i - 1], color=color[i])
        ax[i].grid(True)
        ax[i].set_ylabel('approx')
        ax[i].spines['top'].set_visible(False)

    fig.align_ylabels()
    plt.tight_layout()
    plt.show()


def zad23():
    fig, ax = plt.subplots(3, 2, figsize=(12, 7))
    img = skimage.data.chelsea().astype(np.uint8)
    img = img[:, :, 0]
    vec = np.array([0, 0, 2, 5, 9, 0, -8, -6, 0, 0]).astype(float)
    vec_nor = normalize(vec)
    vec2d = vec_nor[:, None] * vec_nor[None, :]
    vec2d_res = resize(vec2d, (20, 20))
    vec2d_filter = gaussian_filter(vec2d_res, sigma=2)

    img_corr = correlate(img, vec2d_filter)

    ax[0, 0].plot(vec)
    ax[0, 1].plot(vec_nor)
    ax[1, 0].imshow(vec2d)
    ax[1, 1].imshow(vec2d_filter)
    ax[2, 0].imshow(img)
    ax[2, 1].imshow(img_corr)

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(4, 4, figsize=(15, 12))
    s_list = np.linspace(2, 32, 16)
    images = []
    for s in s_list:
        new_wave = resize(vec2d_filter, (s, s))
        new_img = correlate(img, new_wave)
        images.append(new_img)

    for x in range(4):
        for y in range(4):
            ax[x, y].imshow(images[x * 4 + y])
            ax[x, y].set_title(f's = {s_list[x * 4 + y]}')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    zad1()
    zad23()
