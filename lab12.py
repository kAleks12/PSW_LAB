import numpy as np
import skimage.data
import matplotlib.pyplot as plt
from skimage.transform import resize, rotate


def main():
    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    img = skimage.data.camera()
    img = resize(img, (128, 128))
    ax[0, 0].imshow(img, cmap='binary_r')
    gy = np.zeros((128, 128))
    gx = np.zeros((128, 128))

    for x in range(1, 127):
        for y in range(1, 127):
            gx[x, y] = img[x + 1, y] - img[x - 1, y]
            gy[x, y] = img[x, y + 1] - img[x, y - 1]

    ax[0, 1].imshow(gx, cmap='binary_r')
    ax[0, 2].imshow(gy, cmap='binary_r')

    mag = np.sqrt(np.pow(gx, 2) + np.pow(gy, 2))

    ax[1, 0].imshow(mag, cmap='binary_r')

    angle = np.arctan(gy / gx)
    ax[1, 1].imshow(angle, cmap='binary_r')
    plt.show()

    fig, ax = plt.subplots(2, 2, figsize=(12, 7))

    s = 8
    mask = np.zeros_like(img, dtype=int)
    cell_id = 0
    for i in range(0, img.shape[0], s):
        for j in range(0, img.shape[1], s):
            mask[i:i + s, j: j + s] = cell_id
            cell_id += 1

    ax[0, 0].imshow(mask, cmap='seismic')
    bins = 9
    step = np.pi / bins
    hog = np.zeros((256, bins))

    for id in range(cell_id):
        ang_v = angle[mask == id]
        mag_v = mag[mask == id]

        for bin_id in range(bins):
            start = bin_id * step - np.pi / 2
            end = (bin_id + 1) * step - np.pi / 2
            b_mask = (ang_v >= start) * (ang_v < end)
            hog[id, bin_id] = np.sum(mag_v[b_mask])

    hog_org = np.copy(hog)
    hog = hog.reshape((16, 16, 9))
    for i in range(9):
        hog[:, :, i] -= np.min(hog[:, :, i])
        hog[:, :, i] /= np.max(hog[:, :, i])

    channel1 = hog[:, :, 0:3]
    channel2 = hog[:, :, 3:6]
    channel3 = hog[:, :, 6:]

    ax[0, 1].imshow(channel1)
    ax[1, 0].imshow(channel2)
    ax[1, 1].imshow(channel3)
    plt.show()

    angles = np.linspace(-80, 80, bins)
    rec_img = np.zeros((128*128))
    mask_flat = np.reshape(mask, (128*128))
    for i in range(256):
        part = np.zeros((8, 8))
        for j in range(9):
            new_part = np.zeros((8,8))
            new_part[4:5, :] = 1
            angle = angles[j]
            new_part = rotate(new_part, angle)
            new_part = new_part * hog_org[i, j]
            part += new_part
        part = np.reshape(part, (8*8))
        rec_img[mask_flat == i] = part
    rec_img = rec_img.reshape((128, 128))
    plt.imshow(rec_img, cmap='binary_r')
    plt.show()


if __name__ == "__main__":
    main()
