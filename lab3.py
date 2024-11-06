import numpy as np
import matplotlib.pyplot as plt
import skimage as ski


def make_hist_simple(img):
    c_full = np.zeros(256)
    u, c = np.unique(img, return_counts=True)
    c_full[u] = c

    c_prob = c_full / np.sum(c_full)
    return np.linspace(0, 255, 256), c_prob


def make_hist(img):
    ur, cr = np.unique(img[:, :, 0], return_counts=True)
    cr_prob = cr / np.sum(cr)

    ug, cg = np.unique(img[:, :, 1], return_counts=True)
    cg_prob = cg / np.sum(cg)

    ub, cb = np.unique(img[:, :, 2], return_counts=True)
    cb_prob = cb / np.sum(cb)

    return ur, cr_prob, ug, cg_prob, ub, cb_prob


def execute():
    # Values
    D = 8
    L = np.power(2, D).astype(int)

    raw_img = ski.data.chelsea()

    # define luts
    lut_id = np.linspace(0, L - 1, L).astype(int)
    lut_neg = np.linspace(L - 1, 0, L).astype(int)
    lut_step = np.zeros(L).astype(int)
    lut_step[50:150] = L - 1
    lut_sin = np.linspace(0, 2 * np.pi, L)
    lut_sin = (((np.sin(lut_sin) + 1) / 2) * L - 1).astype(int)
    lut_gamma_03 = np.pow(lut_id, 0.3)
    lut_gamma_03 -= np.min(lut_gamma_03)
    lut_gamma_03 /= np.max(lut_gamma_03)
    lut_gamma_03 *= L - 1
    lut_gamma_03 = lut_gamma_03.astype(int)
    lut_gamma_3 = np.pow(lut_id, 3.0)
    lut_gamma_3 -= np.min(lut_gamma_3)
    lut_gamma_3 /= np.max(lut_gamma_3)
    lut_gamma_3 *= L - 1
    lut_gamma_3 = lut_gamma_3.astype(int)

    # transform images
    img_id = lut_id[raw_img]
    img_neg = lut_neg[raw_img]
    img_step = lut_step[raw_img]
    img_sin = lut_sin[raw_img]
    img_gamma_03 = lut_gamma_03[raw_img]
    img_gamma_3 = lut_gamma_3[raw_img]

    # Define objects
    fig, ax = plt.subplots(6, 3, figsize=(12, 12))
    ax[0, 0].scatter(lut_id, lut_id, c='black', s=1)
    ax[0, 1].imshow(img_id, cmap='binary_r')
    xr, yr, xg, yg, xb, yb = make_hist(img_id)
    ax[0, 2].scatter(xr, yr, c='r', s=1)
    ax[0, 2].scatter(xg, yg, c='g', s=1)
    ax[0, 2].scatter(xb, yb, c='b', s=1)

    ax[1, 0].scatter(lut_id, lut_neg, c='black', s=1)
    ax[1, 1].imshow(img_neg, cmap='binary_r')
    xr, yr, xg, yg, xb, yb = make_hist(img_neg)
    ax[1, 2].scatter(xr, yr, c='r', s=1)
    ax[1, 2].scatter(xg, yg, c='g', s=1)
    ax[1, 2].scatter(xb, yb, c='b', s=1)

    ax[2, 0].scatter(lut_id, lut_step, c='black', s=1)
    ax[2, 1].imshow(img_step, cmap='binary_r')
    xr, yr, xg, yg, xb, yb = make_hist(img_step)
    ax[2, 2].scatter(xr, yr, c='r', s=1)
    ax[2, 2].scatter(xg, yg, c='g', s=1)
    ax[2, 2].scatter(xb, yb, c='b', s=1)

    ax[3, 0].scatter(lut_id, lut_sin, c='black', s=1)
    ax[3, 1].imshow(img_sin, cmap='binary_r')
    xr, yr, xg, yg, xb, yb = make_hist(img_sin)
    ax[3, 2].scatter(xr, yr, c='r', s=1)
    ax[3, 2].scatter(xg, yg, c='g', s=1)
    ax[3, 2].scatter(xb, yb, c='b', s=1)

    ax[4, 0].scatter(lut_id, lut_gamma_03, c='black', s=1)
    ax[4, 1].imshow(img_gamma_03, cmap='binary_r')
    xr, yr, xg, yg, xb, yb = make_hist(img_gamma_03)
    ax[4, 2].scatter(xr, yr, c='r', s=1)
    ax[4, 2].scatter(xg, yg, c='g', s=1)
    ax[4, 2].scatter(xb, yb, c='b', s=1)

    ax[5, 0].scatter(lut_id, lut_gamma_3, c='black', s=1)
    ax[5, 1].imshow(img_gamma_3, cmap='binary_r')
    xr, yr, xg, yg, xb, yb = make_hist(img_gamma_3)
    ax[5, 2].scatter(xr, yr, c='r', s=1)
    ax[5, 2].scatter(xg, yg, c='g', s=1)
    ax[5, 2].scatter(xb, yb, c='b', s=1)

    plt.show()

    fig, ax = plt.subplots(2, 3, figsize=(7, 4))
    raw_img = ski.data.moon()
    ax[0, 0].imshow(raw_img, cmap='binary_r')

    x, y = make_hist_simple(raw_img)
    ax[0, 1].bar(x, y)

    y_cum = np.cumsum(y)
    ax[0, 2].scatter(x, y_cum, s=1)

    lut_cum = y_cum * (L - 1)
    lut_cum = lut_cum.astype(int)
    ax[1, 0].scatter(x, lut_cum, s=1)

    img_cum = lut_cum[raw_img]
    ax[1, 1].imshow(img_cum, cmap='binary_r')

    x_cum, y_cum = make_hist_simple(img_cum)
    ax[1, 2].bar(x_cum, y_cum)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    execute()
