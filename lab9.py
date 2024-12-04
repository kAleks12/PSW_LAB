import matplotlib.pyplot as plt
import numpy as np
import skimage.data
from skimage.draw import disk
from random import Random


def get_image():
    img = np.zeros((100, 100))
    np.random.seed(1299)
    for _ in range(100):
        center = (np.random.randint(0, 100), np.random.randint(0, 100))
        rad = np.random.randint(2, 10)
        rr, cc = disk(center, rad, shape=img.shape)
        img[rr, cc] = 1

    return img


def zad1(img):
    fig, ax = plt.subplots(1, 3, figsize=(12, 7))

    img_pad = np.pad(img, pad_width=1)
    B = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    E = np.zeros_like(img)
    for x in range(1, img.shape[0] - 1):
        for y in range(1, img.shape[1] - 1):
            part = img_pad[x - 1:x + 2, y - 1:y + 2]
            out = np.sum((part * B)) == 5
            E[x - 1, y - 1] = out
    diff = img - E
    ax[0].imshow(img, cmap='binary')
    ax[1].imshow(E, cmap='binary')
    ax[2].imshow(diff, cmap='binary')
    plt.tight_layout()
    plt.show()


def zad2(img):
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))

    img_pad = np.pad(img, pad_width=1)
    B = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]])
    B_flip = np.flip(B)

    E = np.zeros_like(img)
    D = np.zeros_like(img)
    for x in range(1, img.shape[0] - 1):
        for y in range(1, img.shape[1] - 1):
            part = img_pad[x - 1:x + 2, y - 1:y + 2]
            out_e = np.sum((part * B)) == 5
            out_d = np.sum((part * B_flip)) >= 1
            E[x - 1, y - 1] = out_e
            D[x - 1, y - 1] = out_d

    O = np.zeros_like(E)
    E_pad = np.pad(E, pad_width=1)
    for x in range(1, E.shape[0]):
        for y in range(1, E.shape[1] - 1):
            part = E_pad[x - 1:x + 2, y - 1:y + 2]
            out = np.sum((part * B_flip)) >= 1
            O[x - 1, y - 1] = out

    C = np.zeros_like(D)
    D_pad = np.pad(D, pad_width=1)
    for x in range(1, D.shape[0] - 1):
        for y in range(1, D.shape[1] - 1):
            part = D_pad[x - 1:x + 2, y - 1:y + 2]
            out = np.sum((part * B)) == 5
            C[x - 1, y - 1] = out

    ax[0, 0].imshow(E, cmap='binary')
    ax[0, 1].imshow(D, cmap='binary')
    ax[1, 0].imshow(O, cmap='binary')
    ax[1, 1].imshow(C, cmap='binary')

    plt.tight_layout()
    plt.show()


def zad3(img):
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))

    img_pad = np.pad(img, pad_width=4)

    B1 = np.zeros((9, 9))
    rr, cc = disk((4, 4), 4, shape=B1.shape)
    B1[rr, cc] = 1

    B2 = np.ones((9, 9))
    B2[0, 0] = 0
    B2[8, 0] = 0
    B2[0, 8] = 0
    B2[8, 8] = 0
    B2[rr, cc] = 0
    np.sum(B1)

    hit_miss = np.zeros_like(img)
    for x in range(4, img.shape[0]):
        for y in range(4, img.shape[1]):
            part = img_pad[x - 4:x + 5, y - 4:y + 5]
            out = ((np.sum((part * B1))) == np.sum(B1)) and (np.sum(((1 - part) * B2)) == np.sum(B2))
            hit_miss[x - 4, y - 4] = out

    ax[0, 0].imshow(B1, cmap='binary')
    ax[0, 1].imshow(B2, cmap='binary')
    ax[1, 0].imshow(img, cmap='binary')
    ax[1, 1].imshow(hit_miss, cmap='binary')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    img = get_image()
    zad1(img)
    zad2(img)
    zad3(img)
