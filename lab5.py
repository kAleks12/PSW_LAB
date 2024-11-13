import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

def normalize(img):
    img -= np.min(img)
    img /= np.max(img)
    return img

def show_fft_results(raw_img):
    # Define objects
    fig, ax = plt.subplots(2, 3, figsize=(12, 7))

    ax[0, 0].imshow(raw_img, cmap='magma')

    img_ft = np.fft.fftshift(np.fft.fft2(raw_img))
    img_imag = np.log(np.abs(img_ft.imag) + 1)
    img_real = np.log(np.abs(img_ft.real) + 1)

    ax[0, 1].imshow(img_real, cmap='magma')
    ax[0, 2].imshow(img_imag, cmap='magma')

    phi = np.arctan2(img_ft.imag,  img_ft.real)
    ax[1, 0].imshow(phi, cmap='magma')

    mag = np.log(np.abs(img_ft) + 1)
    ax[1, 1].imshow(mag, cmap='magma')

    inv_img = np.fft.ifft2(np.fft.ifftshift(img_ft)).real
    ax[1, 2].imshow(inv_img, cmap='magma')

    plt.tight_layout()
    plt.show()


def zad2_img():
    lin = np.linspace(0, 15, 1000)
    x, y = np.meshgrid(lin, lin)

    ampl = [1, 3, 2, 7, 9]
    angl = [np.pi * 1.7, np.pi * 1.5, np.pi * 4, np.pi * 0.75, np.pi * 3]
    wave = [5, 2, 8, 9, 4]

    zero = np.zeros((1000, 1000))

    for params in zip(ampl, angl, wave):
        zero += params[0] * np.sin(2 * np.pi * (x * np.cos(params[1]) + y * np.sin(params[1])) * (1 / params[2]))

    return normalize(zero)


def zad3():
    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    img = ski.data.chelsea()
    img = np.mean(img, axis=2)
    ax[0, 0].imshow(img, cmap='binary_r')

    img_ft = np.fft.fftshift(np.fft.fft2(img))
    mag = np.log(np.abs(img_ft) + 1)
    ax[0, 1].imshow(mag, cmap='magma')

    img_rgb = ski.data.chelsea().astype(float)

    red_canal = np.fft.ifft2(np.fft.ifftshift(img_ft.real)).real
    red_canal = normalize(red_canal)
    img_rgb[:, :, 0] = red_canal
    ax[1, 0].imshow(red_canal, cmap='Reds_r')

    green_canal = np.fft.ifft2(np.fft.ifftshift(img_ft.imag)).imag
    green_canal = normalize(green_canal)
    img_rgb[:, :, 1] = green_canal
    ax[1, 1].imshow(green_canal, cmap='Greens_r')

    blue_canal = np.fft.ifft2(np.fft.ifftshift(img_ft)).real
    blue_canal = normalize(blue_canal)
    img_rgb[:, :, 2] = blue_canal
    ax[1, 2].imshow(blue_canal, cmap='Blues_r')

    ax[0, 2].imshow(img_rgb, cmap='binary_r')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    raw_img = np.zeros((1000, 1000))
    raw_img[500:520, 460:550] = 1

    show_fft_results(raw_img)
    show_fft_results(zad2_img())
    zad3()
