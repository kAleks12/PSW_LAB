import numpy as np
import matplotlib.pyplot as plt


def first_row_data():
    mono = np.zeros((30, 30), int)
    mono[10:20, 10:20] = 1
    mono[15:25, 5:15] = 2
    print(mono)
    return mono


def second_row_data():
    color = np.zeros((30, 30, 3))
    color[15:25, 5:15, 0] = 1
    color[10:20, 10:20, 1] = 1
    color[5:15, 15:25, 2] = 1
    return color, 1 - color


def display_and_save():
    mono = first_row_data()
    fig, ax = plt.subplots(2, 2, figsize=(7, 7))
    ax[0, 0].imshow(mono)
    ax[0, 0].set_title('Plot 1')
    ax[0, 1].imshow(mono, cmap='binary')
    ax[0, 1].set_title('Plot 2')
    color, negative = second_row_data()
    ax[1, 0].imshow(color)
    ax[1, 1].imshow(negative)

    plt.savefig('lab0.png', dpi=fig.dpi)
    plt.show()


if __name__ == '__main__':
    display_and_save()
