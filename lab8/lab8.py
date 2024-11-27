import matplotlib.pyplot as plt
import numpy as np
import skimage.data
from pywt import wavedec
from skimage.transform import resize


def zad1():
    fig, ax = plt.subplots(3, 2, figsize=(12, 7))
    img = skimage.data.chelsea().astype(np.uint8)
    img = np.mean(img, axis=2)
    img = img[1:, :]
    img_ft = np.fft.fftshift(np.fft.fft2(img))
    mag = np.log(np.abs(img_ft) + 1)
    ax[0, 0].imshow(img, cmap='binary_r')
    ax[0, 1].imshow(mag, cmap='magma')

    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_pad = np.pad(kernel,
                        ((int((img.shape[0] - kernel.shape[0]) / 2),
                          int((img.shape[0] - kernel.shape[0]) / 2)),
                         (int((img.shape[1] - kernel.shape[1]) / 2),
                          int((img.shape[1] - kernel.shape[1]) / 2))),
                        mode='constant')
    kernel_ft = np.fft.fftshift(np.fft.fft2(kernel_pad))
    kernel_mag = np.log(np.abs(kernel_ft) + 1)
    ax[1, 0].imshow(kernel_pad, cmap='binary_r')
    ax[1, 1].imshow(kernel_mag, cmap='magma')

    img_fil_ft = img_ft * kernel_ft
    filtered_mag = np.log(np.abs(img_fil_ft) + 1)
    img_fil = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(img_fil_ft))).real

    ax[2, 0].imshow(filtered_mag, cmap='magma')
    ax[2, 1].imshow(img_fil, cmap='binary_r')

    plt.tight_layout()
    plt.show()


def zad2():
    img = plt.imread('filtered.png')
    img = img[:, :, 0]
    filter = np.eye(13, 13)
    filter *= 10
    filter -= 1
    filter /= -39
    filter_pad = np.pad(filter,
                        ((int((img.shape[0] - filter.shape[0]) / 2),
                          int((img.shape[0] - filter.shape[0]) / 2)),
                         (int((img.shape[1] - filter.shape[1]) / 2),
                          int((img.shape[1] - filter.shape[1]) / 2))),
                        mode='constant')

    img_ft = np.fft.fftshift(np.fft.fft2(img))
    mag = np.log(np.abs(img_ft) + 1)

    filter_ft = np.fft.fftshift(np.fft.fft2(filter_pad))
    filter_mag = np.log(np.abs(filter_ft) + 1)

    normal_img_ft = img_ft / filter_ft
    normal_mag = np.log(np.abs(normal_img_ft) + 1)

    normal_img = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(normal_img_ft))).real

    fig, ax = plt.subplots(3, 2, figsize=(12, 7))
    ax[0, 0].imshow(img, cmap='binary_r')
    ax[0, 1].imshow(mag, cmap='magma')
    ax[1, 0].imshow(filter_pad, cmap='binary_r')
    ax[1, 1].imshow(filter_mag, cmap='magma')
    ax[2, 0].imshow(normal_mag, cmap='magma')
    ax[2, 1].imshow(normal_img, cmap='binary_r')

    plt.tight_layout()
    plt.show()


def zad3():
    img = plt.imread('filtered.png')
    img = img[:, :, 0]
    filter = np.eye(13, 13)
    filter *= 10
    filter -= 1
    filter /= -39
    filter_pad = np.pad(filter,
                        ((int((img.shape[0] - filter.shape[0]) / 2),
                          int((img.shape[0] - filter.shape[0]) / 2)),
                         (int((img.shape[1] - filter.shape[1]) / 2),
                          int((img.shape[1] - filter.shape[1]) / 2))),
                        mode='constant')

    img_ft = np.fft.fftshift(np.fft.fft2(img))
    mag = np.log(np.abs(img_ft) + 1)

    filter_ft = np.fft.fftshift(np.fft.fft2(filter_pad))
    filter_mag = np.log(np.abs(filter_ft) + 1)

    normal_img_ft = img_ft / filter_ft
    normal_img_ft *= (1 / (1 + (0.02 / np.pow(filter_ft, 2))))
    normal_mag = np.log(np.abs(normal_img_ft) + 1)
    normal_img = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(normal_img_ft))).real

    fig, ax = plt.subplots(3, 2, figsize=(12, 7))
    ax[0, 0].imshow(img, cmap='binary_r')
    ax[0, 1].imshow(mag, cmap='magma')
    ax[1, 0].imshow(filter_pad, cmap='binary_r')
    ax[1, 1].imshow(filter_mag, cmap='magma')
    ax[2, 0].imshow(normal_mag, cmap='magma')
    ax[2, 1].imshow(normal_img, cmap='binary_r')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    zad1()
    zad2()
    zad3()
