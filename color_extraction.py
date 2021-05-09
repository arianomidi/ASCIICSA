from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def GRAY2HEX(color):
    return "#{0:02x}{0:02x}{0:02x}".format(color)


def get_image(image_path):
    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def getGreyscalePalatte(image, number_of_colors, show_chart=False):
    modified_image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
    # modified_image = modified_image
    # modified_image = modified_image[modified_image > 0]

    print(modified_image)
    modified_image = modified_image.reshape(-1, 1)

    clf = KMeans(n_clusters=number_of_colors)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))

    center_colors = clf.cluster_centers_.flatten()
    center_colors = np.rint(center_colors)
    center_colors = center_colors.astype(int)
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [GRAY2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if show_chart:
        plt.figure(figsize=(10, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap="gray")

        plt.subplot(1, 2, 2)
        plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)

        plt.show()

    return rgb_colors


def get_colors(image, number_of_colors, show_chart=False, greyscale=False):
    modified_image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
    if greyscale:
        modified_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        modified_image = modified_image.reshape(
            modified_image.shape[0] * modified_image.shape[1], 1
        )
    else:
        modified_image = modified_image.reshape(
            modified_image.shape[0] * modified_image.shape[1], 3
        )

    clf = KMeans(n_clusters=number_of_colors)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))

    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    if greyscale:
        hex_colors = [GRAY2HEX(ordered_colors[i]) for i in counts.keys()]
    else:
        hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    print(rgb_colors)

    if show_chart:
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        if greyscale:
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            plt.imshow(gray_img, cmap="gray")
        else:
            plt.imshow(image)

        plt.subplot(1, 2, 2)
        plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)

        plt.show()

    return rgb_colors


def main():
    image = get_image("data/stary_night.jpg")
    get_colors(image, 8, greyscale=True)


if __name__ == "__main__":
    main()
