from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(
        round(color[0]), round(color[1]), round(color[2])
    )


def GRAY2HEX(color):
    return "#{0:02x}{0:02x}{0:02x}".format(round(color))


def GRAY2RGB(brightness):
    brightness_int = round(brightness)
    return (brightness_int, brightness_int, brightness_int)


def getGreyscalePalatte(image, number_of_colors, show_chart=False):
    modified_image = np.array(image)
    modified_image = cv2.resize(
        modified_image, (600, 400), interpolation=cv2.INTER_AREA
    )
    modified_image = modified_image.reshape(-1, 1)

    clf = KMeans(n_clusters=number_of_colors)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))

    center_colors = clf.cluster_centers_.flatten()
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    rgb_colors = [GRAY2RGB(ordered_colors[i]) for i in counts.keys()]

    if show_chart:
        plt.figure(figsize=(10, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap="gray")

        plt.subplot(1, 2, 2)
        hex_colors = [GRAY2HEX(ordered_colors[i]) for i in counts.keys()]
        plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)

        plt.show()

    return rgb_colors


def getColorPalatte(image, number_of_colors, show_chart=False):
    modified_image = np.array(image)
    modified_image = cv2.resize(
        modified_image, (600, 400), interpolation=cv2.INTER_AREA
    )
    modified_image = modified_image.reshape(-1, 3)

    clf = KMeans(n_clusters=number_of_colors)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))

    center_colors = clf.cluster_centers_
    center_colors = np.rint(center_colors)
    center_colors = center_colors.astype(int)
    center_colors = [tuple(color) for color in center_colors]
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if show_chart:
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image)

        plt.subplot(1, 2, 2)
        plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)

        plt.show()

    return rgb_colors
