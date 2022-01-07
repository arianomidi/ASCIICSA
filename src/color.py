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
ansi_palatte = [
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

rgb_palatte = [
    "#ffffff",
    "#ff8f8f",
    "#ff3838",
    "#960000",
    "#460000",
    "#bfffbf",
    "#6aff6a",
    "#00ff00",
    "#009600",
    "#004600",
    "#bfbfff",
    "#6a6aff",
    "#3838ff",
    "#0000ff",
    "#000096",
    "#000046",
    "#000000",
]

rb_palatte = [
    "#000000",
    "#ffdbeb",
    "#ffb2d4",
    "#f76cae",
    "#f72585",
    "#b5179e",
    "#7209b7",
    "#560bad",
    "#03045e",
    "#00013c",
    "#3a0ca3",
    "#3f37c9",
    "#4361ee",
    "#4895ef",
    "#4cc9f0",
    "#95e6ff",
    "#caf3ff",
    "#ffffff",
]

gb_palatte = [
    "#000000",
    "#184e77",
    "#1e6091",
    "#1a759f",
    "#168aad",
    "#34a0a4",
    "#52b69a",
    "#76c893",
    "#99d98c",
    "#b5e48c",
    "#d9ed92",
    "#ffffff",
]

b_palatte = [
    "#000000",
    "#10002b",
    "#240046",
    "#3c096c",
    "#5a189a",
    "#7400b8",
    "#6930c3",
    "#5e60ce",
    "#5390d9",
    "#4ea8de",
    "#48bfe3",
    "#56cfe1",
    "#64dfdf",
    "#72efdd",
    "#80ffdb",
    "#aaffe7",
    "#cafff2",
    "#ffffff",
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


def palette_rgb(colorSceme="ansi16"):
    """Return the selected color palatte (defalut: standard 4 bit ANSI)"""
    if colorSceme == "ansi8":
        palatte_hex = ansi_palatte[0:8]
    elif colorSceme == "ansi16":
        palatte_hex = ansi_palatte[0:16]
    elif colorSceme == "rgb":
        palatte_hex = rgb_palatte
    elif colorSceme == "rb":
        palatte_hex = rb_palatte
    elif colorSceme == "gb":
        palatte_hex = gb_palatte
    elif colorSceme == "b":
        palatte_hex = b_palatte

    palatte_rgb = []

    for hex in palatte_hex:
        hex_code = hex.lstrip("#")
        rgb = tuple(int(hex_code[2 * i : 2 * i + 2], 16) for i in range(3))
        palatte_rgb.append(rgb)

    return palatte_rgb


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
        return ansi_palatte[ansi]

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
        return ansi_palatte[16]
    else:
        return ansi_palatte[17]


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
