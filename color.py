"""
Color utils, including color conversion and color sampling from an image

Author: Arian Omidi
Email: arian.omidi@icloud.com
GitHub: https://github.com/ArianOmidi
Date: 2021-06-01
"""
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter

# Ubuntu default ANSI color scheme
std_rgb = [
    "#2e3436",
    "#cc0000",
    "#4e9a06",
    "#c4a000",
    "#3465a4",
    "#75507b",
    "#06989a",
    "#d3d7cf",
    "#555753",
    "#ef2929",
    "#8ae234",
    "#fce94f",
    "#729fcf",
    "#ad7fa8",
    "#34e2e2",
    "#eeeeec",
    "#000000",
    "#eeeeec",
]

# ======================  COLOR CONVERSIONS ====================== #


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(
        round(color[0]), round(color[1]), round(color[2])
    )


def HEX2RGB(color):
    return (int(color[1:3]), int(color[3:5]), int(color[5:]))


def GRAY2HEX(color):
    return "#{0:02x}{0:02x}{0:02x}".format(round(color))


def GRAY2RGB(brightness):
    brightness_int = round(brightness)
    return (brightness_int, brightness_int, brightness_int)


def RGB2ANSI256(r, g, b):
    # we use the extended greyscale palette here, with the exception of
    # black and white. normal palette only has 4 greyscale shades.
    if r == g and g == b:
        if r < 8:
            ansi_num = 16
        elif r > 248:
            ansi_num = 231
        else:
            ansi_num = round(((r - 8) / 247) * 24) + 232
    else:
        ansi_num = (
            16
            + (36 * round(r / 255 * 5))
            + (6 * round(g / 255 * 5))
            + round(b / 255 * 5)
        )

    ansi_code = "\033[38;5;{}m".format(ansi_num)

    return ansi_code


def HEX2ANSI256(hex):
    return RGB2ANSI256(*HEX2RGB(hex))


# ======================  ANSI METHODS ====================== #


def ansi8_rgb():
    """Return the standard 3 bit ANSI color palatte"""
    ansi8_hex = std_rgb[0:8]
    ansi8_rgb = []

    for hex in ansi8_hex:
        hex_code = hex.lstrip("#")
        rgb = tuple(int(hex_code[2 * i : 2 * i + 2], 16) for i in range(3))
        ansi8_rgb.append(rgb)

    return ansi8_rgb


def ansi16_rgb():
    """Return the standard 4 bit ANSI color palatte"""
    ansi16_hex = std_rgb[0:16]
    ansi16_rgb = []

    for hex in ansi16_hex:
        hex_code = hex.lstrip("#")
        rgb = tuple(int(hex_code[2 * i : 2 * i + 2], 16) for i in range(3))
        ansi16_rgb.append(rgb)

    return ansi16_rgb


def parse_ansi(ansi):
    """
    Takes raw ansi code and returns 8-bit color code.
    """
    _, raw_ansi = ansi.split("[")
    codes = raw_ansi.split(";")

    if codes[0] == "38" and codes[1] == "5":  # if 8-bit ansi color code return color
        return int(codes[-1][:-1])
    else:
        end_code = int(codes[-1][:-1])
        if (
            30 <= end_code and end_code <= 37
        ):  # if 3-bit standard ansi color code return 8-bit color code
            return end_code - 30
        elif (
            90 <= end_code and end_code <= 97
        ):  # if 3-bit bright ansi color code return 8-bit color code
            return end_code - 82
        else:
            return -1


def ANSI2RGB(ansi):
    """Convert raw ansi code and hexcolor code"""
    ansi = parse_ansi(ansi)

    if ansi < 0 or ansi > 255:
        return "#000000"
    if ansi < 16:
        return std_rgb[ansi]

    if ansi > 231:
        s = (ansi - 232) * 10 + 8
        rgb = "#%02x%02x%02x" % (s, s, s)
        return rgb

    index_R = (ansi - 16) // 36
    r = 55 + index_R * 40 if index_R > 0 else 0
    index_G = ((ansi - 16) % 36) // 6
    g = 55 + index_G * 40 if index_G > 0 else 0
    index_B = (ansi - 16) % 6
    b = 55 + index_B * 40 if index_B > 0 else 0

    rgb = "#%02x%02x%02x" % (r, g, b)
    return rgb


def background_color(inverted):
    """Return background color based on if the image is inverted"""
    if inverted:
        return std_rgb[16]
    else:
        return std_rgb[17]


# ======================  COLOR SAMPLING METHODS  ====================== #


def getGreyscalePalatte(image, num, show_chart=False):
    """
    Returns the most prevelent shades of a greyscale image

    arguments:
    image - image to sample colors from
    num - number of shades to sample
    show_chart - show a visual representation of the shades selected
    """
    modified_image = np.array(image)
    modified_image = cv2.resize(
        modified_image, (600, 400), interpolation=cv2.INTER_AREA
    )
    modified_image = modified_image.reshape(-1, 1)

    clf = KMeans(n_clusters=num)
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


def getColorPalatte(image, num, show_chart=False):
    """
    Returns the most prevelent colors of an image

    arguments:
    image - image to sample colors from
    num - number of colors to sample
    show_chart - show a visual representation of the colors selected
    """
    modified_image = np.array(image)
    modified_image = cv2.resize(
        modified_image, (600, 400), interpolation=cv2.INTER_AREA
    )
    modified_image = modified_image.reshape(-1, 3)

    clf = KMeans(n_clusters=num)
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
