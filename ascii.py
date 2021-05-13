# Python code to convert an image to ASCII image.
import sys, argparse, time
import numpy as np
from scipy.ndimage.filters import gaussian_filter, percentile_filter
from scipy import ndimage
import math
from pathlib import Path
import matplotlib.pyplot as plt
from colorthief import ColorThief

from PIL import Image, ImageFilter, ImageFont, ImageDraw, ImageOps
import cv2
from skimage.measure import block_reduce
from skimage.transform import downscale_local_mean, resize, rescale


from ansi import *
from color_extraction import *

# gray scale level values from:
# http://paulbourke.net/dataformats/asciiart/

# 70 levels of gray
gscale1 = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "

# 10 levels of gray
gscale2 = "@%#*+=-:. "
# gscale2 = "ARIANomidi"

# color codes schemes
color_schemes = {
    "binary": ["\033[37m"],
    "grayscale_4b": ["\033[0m", "\033[90m", "\033[37m", "\033[97m"],
    "grayscale_3": ["\033[0m", "\033[38;5;241m", "\033[38;5;248m", "\033[38;5;255m"],
    "grayscale_5": [
        "\033[38;5;234m",
        "\033[38;5;235m",
        "\033[38;5;239m",
        "\033[38;5;244m",
        "\033[38;5;250m",
        "\033[38;5;255m",
    ],
    "rgb_colorful": ["\033[0m", "\033[34m", "\033[31m", "\033[33m", "\033[37m"],
    "rgb_cool": ["\033[0m", "\033[34m", "\033[35m", "\033[95m", "\033[37m"],
}


def normalizeTiles(tiles):
    """
    Given numpy array, filter and return normalized image array
    """
    """print(tiles.min(), tiles.max())"""

    # percentage filter
    min_val = np.percentile(tiles, 0)
    max_val = np.percentile(tiles, 100)
    tiles = np.clip(tiles, min_val, max_val)

    """print(tiles.min(), tiles.max())"""

    # normalize tile value array
    normalized_tiles = (tiles - min_val) / (max_val - min_val)

    return normalized_tiles


def filterImage(image):
    print(image.min(), image.max())

    # percentage filter
    min_val = np.percentile(image, 0)
    max_val = np.percentile(image, 100)
    filtered_img = np.clip(image, min_val, max_val)

    print(image.min(), image.max())

    # normalize tile value array
    normalized_img = (filtered_img - min_val) / (max_val - min_val)

    return normalized_img

    # return image.filter(ImageFilter.DETAIL)

    # ------ todo: remove ------
    # f = plt.figure()
    # f.add_subplot(1,2, 1)
    # plt.imshow(im, cmap='gray', vmin=0, vmax=255)

    # im = ndimage.median_filter(im, 3)

    # f.add_subplot(1, 2, 2)
    # plt.imshow(block_mean, cmap='gray', vmin=0, vmax=255)
    # plt.show(block=True)
    # ------ todo: remove ------


def autoColor(image, colorCount, greyscale=False, show_chart=False):
    # TODO: refactor
    if greyscale:
        image = image.convert("L")

    image_cv = np.array(image)

    if greyscale:
        colors = getGreyscalePalatte(image_cv, colorCount, show_chart)
    else:
        colors = getColorPalatte(image_cv, colorCount, show_chart)

    # colors.sort()
    # colors = list(map(lambda val: rgbToAnsi256(val, val, val), colors_grey))

    print("Colors: {}".format(colors))

    return colors


def covertImageToAscii(image, colors, cols, scale, moreLevels, invertImg):
    """
    Given Image and dims (rows, cols) returns an m*n list of Images
    """
    # declare globals
    global gscale1, gscale2

    # store dimensions
    W, H = image.size[0], image.size[1]
    print(" - input image dims: %d x %d" % (W, H))

    # compute width of tile
    w = W / cols

    # compute tile height based on aspect ratio and scale
    h = w / scale

    # compute number of rows
    rows = int(H / h)  # TODO
    w = int(w)
    h = int(h)

    print(" - cols: %f, rows: %f" % (cols, rows))
    print(" - tile dims: %f x %f" % (w, h))

    # check if image size is too small
    if cols > W or rows > H:
        print("Image too small for specified cols!")
        exit(0)

    # get image as numpy array
    # image_greyscale = image.convert("L")
    im = np.array(image)

    print("Block Reduction...")
    start = time.perf_counter()
    r = block_reduce(im[:, :, 0], (h, w), np.median)
    g = block_reduce(im[:, :, 1], (h, w), np.median)
    b = block_reduce(im[:, :, 2], (h, w), np.median)
    tiles_rgb = np.stack((r, g, b), axis=-1)
    print(tiles_rgb.shape)
    tiles_greyscale = np.average(tiles_rgb, axis=2, weights=[0.299, 0.587, 0.114])
    tiles_rgb = np.rint(tiles_rgb).astype(int)
    tiles = normalizeTiles(tiles_greyscale)
    end = time.perf_counter()
    print(f"Completed {end - start:0.4f} seconds")

    # print("Mean Downscale...")
    # start = time.perf_counter()
    # r = downscale_local_mean(im[:, :, 0], (h, w))
    # g = downscale_local_mean(im[:, :, 1], (h, w))
    # b = downscale_local_mean(im[:, :, 2], (h, w))
    # tiles_rgb = np.stack((r, g, b), axis=-1)
    # print(tiles_rgb.shape)
    # tiles_greyscale = np.average(tiles_rgb, axis=2, weights=[0.299, 0.587, 0.114])
    # tiles_rgb = np.rint(tiles_rgb).astype(int)
    # tiles = normalizeTiles(tiles_greyscale)
    # end = time.perf_counter()
    # print(f"Completed {end - start:0.4f} seconds")

    # print("OpenCV sampling...")
    # start = time.perf_counter()
    # tiles_rgb = cv2.resize(im, (cols, rows), interpolation=cv2.INTER_AREA)
    # print(tiles_rgb.shape)
    # tiles_greyscale = np.average(tiles_rgb, axis=2, weights=[0.299, 0.587, 0.114])
    # tiles_rgb = np.rint(tiles_rgb).astype(int)
    # tiles = normalizeTiles(tiles_greyscale)
    # end = time.perf_counter()
    # print(f"Completed {end - start:0.4f} seconds")

    # apply inversion
    if invertImg:
        tiles = 1 - tiles
        colors = list(reversed(colors))

    color_palette = np.asarray(colors)

    # ascii image is a list of character strings
    aimg = []
    # color image is a list of ansi color codes strings
    cimg = []
    # generate list of dimensions
    for j in range(rows):
        # append an empty string
        aimg.append("")
        cimg.append([])

        for i in range(cols):
            # get character representation of tile
            gsval = ""
            char_ratio = tiles[j][i]

            if moreLevels:
                gsval += gscale1[round(69 * char_ratio)]
            else:
                gsval += gscale2[round(9 * char_ratio)]

            # get closest color in palette to that of the tile
            tile_rgb = tiles_rgb[j][i]
            deltas = color_palette - tile_rgb
            dist_2 = np.einsum("ij,ij->i", deltas, deltas)
            color_index = np.argmin(dist_2)

            cimg[j].append(colors[color_index])

            # append ascii char to string
            aimg[j] += gsval

    # return txt image
    return aimg, cimg


def block_mean(ar, fact):
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy / fact * (X / fact) + Y / fact
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx / fact, sy / fact)
    return res


# --------- TODO: refactor -------------- #

WIDTH_SCALING_FACTOR = 0.75
HEIGHT_SCALING_FACTOR = 8 / 11


def text_image(aimg, cimg, inverted, font_path=None):
    """Convert text file to a grayscale image with black characters on a white background.

    arguments:
    text_path - the content of this file will be converted to an image
    font_path - path to a font file (for example impact.ttf)
    """
    # declare globals
    global WIDTH_SCALING_FACTOR, HEIGHT_SCALING_FACTOR

    # # parse the file into lines
    # with open(text_path) as text_file:  # can throw FileNotFoundError
    #     lines = tuple(l.rstrip() for l in text_file.readlines())
    lines = aimg

    # choose a font (you can see more detail in my library on github)
    large_font = 20  # get better resolution with larger size
    default_font_path = (
        "fonts/SFMono-Semibold.otf" if inverted else "fonts/SFMono-Heavy.otf"
    )
    font_path = (
        font_path or default_font_path
    )  # Courier New. works in windows. linux may need more explicit path
    try:
        font = ImageFont.truetype(font_path, size=large_font)

    except IOError:
        font = ImageFont.load_default()
        print("Could not use chosen font. Using default.")

    # make the background image based on the combination of font and lines
    pt2px = lambda pt: int(
        round(pt * 96.0 / 72)
    )  # function that converts points to pixels
    max_width_line = max(
        lines, key=lambda s: font.getsize(s)[0]
    )  # get line with largest width
    # max height is adjusted down because it's too large visually for spacing
    test_string = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnop.,;:'/|{}[]()&$%#@"
    max_height = pt2px(font.getsize(test_string)[1])
    max_width = pt2px(font.getsize(max_width_line)[0])
    char_width = round(max_width / len(max_width_line))
    height = int(
        (max_height) * len(lines) * HEIGHT_SCALING_FACTOR
    )  # perfect or a little oversized
    width = int(max_width * WIDTH_SCALING_FACTOR)  # a little oversized
    image = Image.new("RGB", (width, height), color=background_color(inverted))
    draw = ImageDraw.Draw(image)

    # draw each line of text
    vertical_position = 0
    horizontal_position = 0
    line_spacing = round(
        max_height * HEIGHT_SCALING_FACTOR
    )  # reduced spacing seems better
    char_spacing = round(char_width * WIDTH_SCALING_FACTOR)

    for line, line_colors in zip(lines, cimg):
        hor_pos = horizontal_position
        for c, color in zip(line, line_colors):
            # rbg_color = ansi_rgb(color)
            rbg_color = color
            draw.text((hor_pos, vertical_position), c, fill=rbg_color, font=font)
            hor_pos += char_spacing
        vertical_position += line_spacing

    return image


# ---------------------------------------- #


def convertImageToAscii(
    image,
    colors=color_schemes["grayscale_5"],
    cols=80,
    scale=0.5,
    moreLevels=True,
    invertImg=True,
):
    """
    Converts given image to an ASCII image
    """
    # get text and ANSI colors of image
    aimg, cimg = covertImageToAscii(image, colors, cols, scale, moreLevels, invertImg)
    # convert to image
    return text_image(aimg, cimg, invertImg)


# main() function
def main():
    # create parser
    descStr = "This program converts an image into ASCII art."
    parser = argparse.ArgumentParser(description=descStr)
    # add expected arguments
    parser.add_argument("filename")

    parser.add_argument(
        "-g",
        "--greyscale",
        dest="greyscaleScheme",
        nargs="?",
        type=int,
        const=8,
        default=8,
    )
    parser.add_argument(
        "-c", "--color", dest="colorScheme", type=int, nargs="?", const=8
    )
    parser.add_argument("-a", "--autoColor", dest="autoColor", action="store_true")
    parser.add_argument("-n", "--cols", dest="cols", type=int, required=False)
    parser.add_argument("-l", "--scale", dest="scale", type=float, required=False)
    parser.add_argument("-m", "--morelevels", dest="moreLevels", action="store_true")
    parser.add_argument("-i", "--invert", dest="invert", action="store_false")

    parser.add_argument("-o", "--out", dest="outFile", required=False)
    parser.add_argument("-O", "--imgout", dest="imgOutFile", required=False)
    parser.add_argument("-H", "--hide", dest="hide", action="store_true")
    parser.add_argument("-s", "--save", dest="save", action="store_true")
    parser.add_argument("-p", "--print", dest="print", action="store_true")

    # parse args
    args = parser.parse_args()

    # open image and convert to grayscale
    filename = Path(args.filename)
    if filename.exists():
        image = Image.open(args.filename).convert("RGB")
    else:
        print("ERROR: Image does not exist.")
        exit(0)

    # set text output file
    outFile = "out.txt"
    if args.outFile:
        outFile = args.outFile

    # set img output file
    imgOutFile = Path("out/{}_ascii.png".format(filename.stem))
    if args.imgOutFile:
        imgOutFile = Path(args.imgOutFile)

    # set scale default as 0.5 which suits
    # a Courier font
    scale = 0.5
    if args.scale:
        scale = float(args.scale)

    # set cols
    cols = 80
    if args.cols:
        cols = int(args.cols)

    # set color scheme
    if args.autoColor:
        # TODO: remove timers
        print("generating colors...")
        start = time.perf_counter()
        print(args.colorScheme)

        num_of_colors = args.colorScheme or args.greyscaleScheme
        is_greyscale = args.colorScheme == None
        colors = autoColor(image, num_of_colors, greyscale=is_greyscale)

        end = time.perf_counter()
        print(f"Completed {end - start:0.4f} seconds")
    else:
        if args.colorScheme:
            colors = ansi16_rgb()
        else:
            colors = []
            for i in range(args.greyscaleScheme):
                color = int(i * 255 / (args.greyscaleScheme - 1))
                colors.append((color, color, color))
        print(colors)

    # -------------------------------------- #
    print("generating ASCII art...")
    start = time.perf_counter()

    # convert image to ascii txt
    aimg, cimg = covertImageToAscii(
        image, colors, cols, scale, args.moreLevels, args.invert
    )

    # make image from text
    image = text_image(aimg, cimg, args.invert)

    # # TODO: write to text file
    # f = open(outFile, "w")
    # background_color = "\033[40m" if args.invert else "\033[107m"
    # for line, line_colors in zip(aimg, cimg):
    #     f.write(background_color)
    #     for c, color in zip(line, line_colors):
    #         f.write(color + c)
    #     f.write("\033[0m\n")
    # f.close()

    # # print output
    # if args.print:
    #     f = open(outFile, "r")
    #     print(f.read(), end="")
    #     f.close()

    # save/show the image
    if not args.hide:
        image.show()
    if args.save or args.imgOutFile:
        # Confirm if about to overwrite a file
        if imgOutFile.is_file():
            response = input(
                "Warning: file '{}' already exists.\nContinue and overwrite this file? (y/n) ".format(
                    imgOutFile
                )
            )
            if response.upper() != "Y":
                exit(0)
        image.save(imgOutFile)

    end = time.perf_counter()
    print(f"Completed {end - start:0.4f} seconds")


# call main
if __name__ == "__main__":
    main()
