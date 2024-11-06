import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from scipy.ndimage import correlate, convolve


def convolute_kernel3(img, kernel):
    conv_kernel = np.flip(kernel, (0, 1))
    return correlate_kernel3(img, conv_kernel)


def correlate_kernel3(img, kernel):
    img_x, img_y = img.shape
    kernel_x, kernel_y = kernel.shape

    img_pad = np.pad(img, 1, 'edge')
    new_img = np.zeros(img.shape)

    for x in range(img_x - kernel_x + 1):
        for y in range(img_y - kernel_y + 1):
            new_img[x][y] = np.sum(img_pad[x:x + kernel_x, y: y + kernel_y] * kernel)

    return new_img


def correlate_kerneln(img, kernel):
    img_x, img_y = img.shape
    kernel_x, kernel_y = kernel.shape

    pad = int((kernel_x - 1) / 2)
    img_pad = np.pad(img, pad, 'edge')
    new_img = np.zeros(img.shape)
    for x in range(pad, img_x + pad):
        for y in range(pad, img_y + pad):
            part = img_pad[x-pad:x + pad + 1, y - pad: y + pad + 1]
            new_img[x-pad][y-pad] = np.sum(part * kernel)
    return new_img


def execute_conv_corr():
    raw_img = ski.data.chelsea()
    raw_img = np.mean(raw_img[::, ::], 2)
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))

    s1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    conv_s1 = convolve(raw_img, s1)
    corr_s1 = correlate_kernel3(raw_img, s1)
    ax[0, 0].imshow(corr_s1, cmap='binary_r')
    ax[0, 0].set_title("Korelacja scipy")

    ax[0, 1].set_title("Konwolucja scipy")
    ax[0, 1].imshow(conv_s1, cmap='binary_r')

    my_corr_s1 = correlate_kernel3(raw_img, s1)
    ax[1, 0].set_title("Korelacja własna")
    ax[1, 0].imshow(my_corr_s1, cmap='binary_r')

    my_conv_s1 = convolute_kernel3(raw_img, s1)
    ax[1, 1].set_title("Konwolucja własna")
    ax[1, 1].imshow(my_conv_s1, cmap='binary_r')

    plt.tight_layout()
    plt.show()


def execute_filter():
    raw_img = ski.data.chelsea()
    raw_img = np.mean(raw_img[::, ::], 2)
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))

    ax[0, 0].set_title("Oryginał")
    ax[0, 0].imshow(raw_img, vmin=0, vmax=255, cmap='binary_r')

    blur_kernel = np.ones((7, 7))
    blur_kernel /= np.sum(blur_kernel)
    blur_img = correlate_kerneln(raw_img, blur_kernel)
    ax[0, 1].set_title("Rozmyty obraz")
    ax[0, 1].imshow(blur_img, vmin=0, vmax=255, cmap='binary_r')

    mask = raw_img - blur_img
    ax[1, 0].set_title('Maska')
    ax[1, 0].imshow(mask, cmap='binary_r')

    retr_img = blur_img + mask
    ax[1, 1].set_title("Unmasked")
    ax[1, 1].imshow(retr_img, vmin=0, vmax=255, cmap='binary_r')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # execute_conv_corr()
    execute_filter()
