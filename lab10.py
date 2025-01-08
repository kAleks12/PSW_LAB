import matplotlib.pyplot as plt
import numpy as np
import skimage.data
from scipy.ndimage import gaussian_filter, correlate, median_filter


def normalize(img):
    img -= np.min(img)
    img /= np.max(img)
    return img


def make_hist_simple(img):
    c_full = np.zeros(256)
    u, c = np.unique(img, return_counts=True)
    c_full[u] = c

    return np.linspace(0, 255, 256), c_full


def main():
    fig, ax = plt.subplots(1, 3, figsize=(12, 7))
    img = skimage.data.chelsea().astype(np.uint8)
    img = np.mean(img, axis=2)
    ax[0].imshow(img, cmap='binary_r')

    img_fil = gaussian_filter(img, sigma=1)
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    img_lapl = correlate(img_fil, kernel)
    img_lapl[img_lapl < 0] = 0
    ax[1].imshow(img_lapl, cmap='binary')

    px = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    py = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    prewitt_x = correlate(img_fil, px)
    prewitt_y = correlate(img_fil, py)
    prewitt_x[prewitt_x < 0] = 0
    prewitt_y[prewitt_y < 0] = 0
    img_gradient = np.abs(prewitt_x) + np.abs(prewitt_y)

    ax[2].imshow(img_gradient, cmap='binary')

    plt.tight_layout()
    plt.show()

    lapl_norm = normalize(img_lapl)
    gradient_norm = normalize(img_gradient)
    fig, ax = plt.subplots(2, 4, figsize=(12, 7))
    ax[0, 0].imshow(img_lapl, cmap='binary')
    ax[1, 0].imshow(img_gradient, cmap='binary')

    lapl_thresh = lapl_norm > 0.1
    gradient_thresh = gradient_norm > 0.1
    ax[0, 1].imshow(lapl_thresh, cmap='binary')
    ax[1, 1].imshow(gradient_thresh, cmap='binary')

    lapl_med_fil = median_filter(lapl_norm, size=(21, 21))
    gradient_med_fil = median_filter(gradient_norm, size=(21, 21))
    ax[0, 2].imshow(lapl_med_fil, cmap='binary')
    ax[1, 2].imshow(gradient_med_fil, cmap='binary')

    lapl_thresh_adapt = lapl_norm > lapl_med_fil
    gradient_thresh_adapt = gradient_norm > gradient_med_fil
    ax[0, 3].imshow(lapl_thresh_adapt, cmap='binary')
    ax[1, 3].imshow(gradient_thresh_adapt, cmap='binary')

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    img_nor = normalize(img_lapl)
    img_dig = (img_nor * 255).astype(int)
    ax[0, 0].imshow(img_dig, cmap='binary')

    x_cum, y_cum = make_hist_simple(img_dig)
    ax[0, 1].bar(x_cum, y_cum)
    ax[0, 1].set_yscale('log')

    results = []
    mean_img = np.mean(img_dig)
    for i in range(255):
        class1 = img_dig[img_dig >= i]
        class0 = img_dig[img_dig < i]
        if np.size(class0) == 0 or np.size(class1) == 0:
            results.append(0)
        else:
            p0 = np.size(class0) / np.size(img_dig)
            p1 = np.size(class1) / np.size(img_dig)
            result = p0 * np.pow((np.mean(class0) - mean_img), 2) + p1 * np.pow((np.mean(class1) - mean_img), 2)
            results.append(result)

    opt_threshold = np.argmax(results)
    ax[1, 0].plot(np.linspace(0, 255, 255), results)
    ax[1, 0].vlines(opt_threshold, ymin=0, ymax=max(results), color='red', linestyles='dashed')

    opt_img = img_dig >= opt_threshold
    ax[1, 1].imshow(opt_img, cmap='binary')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
