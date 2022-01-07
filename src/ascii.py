"""
A program that converts an image into ASCII art v2.0

Author: Arian Omidi
Email: arian.omidi@icloud.com
GitHub: https://github.com/ArianOmidi
Date: 2021-06-01
"""
import os, sys, subprocess
from pathlib import Path
import argparse, time

import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from skimage.measure import block_reduce
from skimage.transform import downscale_local_mean

from color import *
from char_density import *

from math import sqrt
import colorsys

# sampling methods
OPENCV_RESIZE = 0
BLOCK_REDUCE = 1
DOWNSCALE_MEAN_REDUCE = 2

# color selection methods
NEAREST = 0
FIXED = 1

# image filter methods
CONTRAST = 0
BRIGHTNESS = 1


# ======================  HELPER METHODS  ====================== #


def autoSize(image, resolution=1920):
    """Determines the size of an image by setting the largest side to the size of the resolution"""
    if image.width >= image.height:
        size = (resolution, round((resolution / image.width) * image.height))
    else:
        size = (round((resolution / image.height) * image.width), resolution)
    return size


def filterImage(image, factor, type=CONTRAST):
    # select image enhancer
    if type == CONTRAST:
        enhancer = ImageEnhance.Contrast(image)
    else:
        enhancer = ImageEnhance.Brightness(image)

    # return filtered image
    return enhancer.enhance(factor)


def defaultPalatte(shadeCount=8, colorSceme=None):
    if colorSceme:
        palatte = palette_rgb(colorSceme)
    else:
        if shadeCount == 1:
            return [(255, 255, 255)]

        palatte = []
        for i in range(shadeCount):
            color = int(i * 255 / (shadeCount - 1))
            palatte.append((color, color, color))

    return palatte


def autoColor(image, colorCount, greyscale=False, show_chart=False):
    if greyscale:
        image = image.convert("L")
        return getGreyscalePalatte(image, colorCount, show_chart)
    else:
        return getColorPalatte(image, colorCount, show_chart)


def normalizeTiles(tiles):
    """
    Given numpy array, filter and return normalized image array
    """

    # percentage filter
    min_val = np.min(tiles)
    max_val = np.max(tiles)
    tiles = np.clip(tiles, min_val, max_val)

    # normalize tile value array
    if max_val != min_val:
        normalized_tiles = (tiles - min_val) / (max_val - min_val)
    else:
        normalized_tiles = tiles - min_val

    return normalized_tiles


# ======================  ASCII CONVERSION METHODS  ====================== #
def step(r, g, b, repetitions=1):
    lum = sqrt(0.241 * (r) + 0.691 * g + 0.068 * b)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h2 = int(h * repetitions)
    lum2 = int(lum * repetitions)
    v2 = int(v * repetitions)
    return (h2, lum, v2)


def relativeLuminance(r, g, b):
    r_norm = r / 255
    g_norm = g / 255
    b_norm = b / 255

    if r_norm <= 0.03928:
        r_out = r_norm / 12.92
    else:
        r_out = ((r_norm + 0.055) / 1.055) ** 2.4

    if g_norm <= 0.03928:
        g_out = g_norm / 12.92
    else:
        g_out = ((g_norm + 0.055) / 1.055) ** 2.4

    if b_norm <= 0.03928:
        b_out = b_norm / 12.92
    else:
        b_out = ((b_norm + 0.055) / 1.055) ** 2.4

    return 0.2126 * r_out + 0.7152 * g_out + 0.0722 * b_out


def covertImageToAscii(
    image,
    colors,
    cols,
    scale,
    chars,
    invert,
    sampling=OPENCV_RESIZE,
    colorSelection=NEAREST,
):
    """
    Converts given image to array of characters and colors corresponding to the image.

    arguments:
    colors - the color pallate to be used as an RGB array
    cols - the width of the output image in number of letters
    scale - the ratio of rows to cols
    chars - the characters used in the ASCII image
    invert - if the backgroung should be black or white
    sampling - sampling method used to detemine chars (default: RESIZE)
    colorSelection - sampling method used to detemine colors (default: NEAREST)
    """
    # store dimensions
    W, H = image.size[0], image.size[1]
    # compute number of rows
    rows = int(H * scale * cols / W)
    # compute width of tile
    w = int(W / cols)
    # compute tile height based on aspect ratio and scale
    h = int(W / (scale * cols))

    # check if image size is too small
    if cols > W or rows > H:
        print("Image too small for specified cols!")
        exit(0)

    # get image as numpy array
    im = np.array(image)

    # sample the image into desired shape
    if sampling == OPENCV_RESIZE:
        tiles_rgb = cv2.resize(im, (cols, rows), interpolation=cv2.INTER_AREA)
    else:
        if sampling == BLOCK_REDUCE:
            sampling_func = lambda im, size, func: block_reduce(im, size, func)
        else:
            sampling_func = lambda im, size, func: downscale_local_mean(im, size)

        r = sampling_func(im[:, :, 0], (h, w), np.median)
        g = sampling_func(im[:, :, 1], (h, w), np.median)
        b = sampling_func(im[:, :, 2], (h, w), np.median)
        tiles_rgb = np.stack((r, g, b), axis=-1)

    tiles_greyscale = np.average(tiles_rgb, axis=2, weights=[0.299, 0.587, 0.114])
    tiles_rgb = np.rint(tiles_rgb).astype(int)
    tiles = normalizeTiles(tiles_greyscale)

    # apply inversion
    if invert:
        tiles = 1 - tiles
        colors = list(reversed(colors))

    # init color selection
    if colorSelection == NEAREST:
        color_palette = np.asarray(colors)
    else:
        colors.sort(
            key=lambda rgb: relativeLuminance(*rgb),  # relative luminance
            # key=lambda rgb: (
            #     sqrt(
            #         0.299 * (rgb[0] ** 2)
            #         + 0.587 * (rgb[1] ** 2)
            #         + 0.114 * (rgb[2] ** 2)
            #     )
            # ),  # HSP Color Model
            reverse=invert,
        )
        # colors.sort(key=lambda rgb: colorsys.rgb_to_hls(*rgb))
        # colors.sort(key=lambda rgb: step(*rgb, 8), reverse=invert)  # rainbow inverse
        # colors.sort(key=lambda rgb: step(*rgb, 8))  # rainbow

    # ascii image is a list of character strings
    aimg = []
    # color image is a list of ansi color codes strings
    cimg = []
    # generate list of dimensions
    for j in range(rows):
        # append an empty string
        aimg.append([])
        cimg.append([])

        for i in range(cols):
            # get character representation of tile
            char = chars[round((len(chars) - 1) * tiles[j][i])]
            aimg[j].append(char)

            # get color for char of the tile
            if colorSelection == NEAREST:
                # get closest color in palette to that of the tile
                deltas = color_palette - tiles_rgb[j][i]
                dist_2 = np.einsum("ij,ij->i", deltas, deltas)
                color_index = np.argmin(dist_2)
                cimg[j].append(colors[color_index])
            else:
                color_index = round((len(colors) - 1) * tiles[j][i])
                cimg[j].append(colors[color_index])

    return aimg, cimg


def textToImage(aimg, cimg, inverted, size, bg_color=None, font_path=None):
    """
    Convert character and color arrays to an ASCII image.

    arguments:
    aimg - the array of characters to be converted to an image
    cimg - the array of colors coresponding to a char in aimg
    inverted - if the backgroung should be black or white
    size - size of the outputed image
    bg_color - color of background (set to default based on the value of 'inverted')
    font_path - path to a font file (for example impact.ttf)
    """

    # choose a font
    if inverted:
        default_font_path = "../fonts/SFMono-Medium.otf"
    else:
        default_font_path = "../fonts/SFMono-Heavy.otf"
    font_path = font_path or default_font_path
    font_size = round((3 / 4) * (2 * size[1] / len(aimg)))
    try:
        font = ImageFont.truetype(font_path, size=font_size)
    except IOError:
        font = ImageFont.load_default()
        print("Warning: Could not use chosen font. Using default.")

    # char height is adjusted based on output size and col:row ratio
    line_width = font.getsize("".join(aimg[0]))[0]  # get line width
    char_width = round(line_width / len(aimg[0]))
    char_height = round((size[1] / size[0]) * (line_width / len(aimg)))

    # create new image
    height = round(char_height * len(aimg))
    width = round(line_width)
    bg_color = bg_color or background_color(inverted)
    image = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(image)

    # draw each line of text
    vert_pos = 0
    line_spacing = char_height
    char_spacing = char_width

    # draw each char to image
    for line, line_colors in zip(aimg, cimg):
        hor_pos = 0
        for c, color in zip(line, line_colors):
            draw.text((hor_pos, vert_pos), c, fill=color, font=font)
            hor_pos += char_spacing
        vert_pos += line_spacing

    return image.resize(size)


def convertImageToAscii(
    image,
    colors=defaultPalatte(),
    cols=120,
    scale=0.6,
    chars=ascii_chars,
    invert=True,
    filterType=CONTRAST,
    filterFactor=1.3,
    size=None,
    bg_color=None,
    font=None,
):
    """
    Converts given image to an ASCII image (performs covertImageToAscii and then textToImage)

    optional arguments:
    colors - the color pallate to be used as an RGB array
    cols - the width of the output image in number of letters
    scale - the ratio of rows to cols
    chars - the characters used in the ASCII image
    invert - if the backgroung should be black or white
    filterType - type of filter (CONTRAST or BRIGHTNESS)
    filterFactor - the degree the filter will be applied (factor of 1 is equivelent to no filter)
    size - size of the outputed image
    bg_color - color of background (set to default based on the value of 'invert')
    font - path to a font file (for example impact.ttf)
    """
    image = filterImage(image, filterFactor, type=filterType)
    if not size:
        size = autoSize(image)

    # get text and ANSI colors of image
    aimg, cimg = covertImageToAscii(image, colors, cols, scale, chars, invert)

    # convert to image
    return textToImage(aimg, cimg, invert, size, bg_color=bg_color, font_path=font)


# ======================  MAIN PROGRAM  ====================== #


def main():
    # create parser
    descStr = "This program converts an image into ASCII art."
    parser = argparse.ArgumentParser(description=descStr)
    # add expected arguments
    parser.add_argument("filename", help="Path to image to be converted.")

    ### COLOR ARGS ###

    parser.add_argument(
        "-g",
        "--greyscale",
        action="store_true",
        help="Select for greyscale image and pass number of shades used (defaults to true and 8 shades).",
    )
    parser.add_argument(
        "-gs",
        "--greyscaleSamples",
        type=int,
        default=8,
        help="Number of samples in palette when in greyscale mode (defalut: 8).",
    )
    parser.add_argument(
        "-c",
        "--colorPalatte",
        type=str.lower,
        choices=["ansi8", "ansi16", "rgb", "rb", "gb", "b"],
        const="ansi16",
        nargs="?",
        help="Select color palatte used to for colored image (default: ansi16).",
    )
    parser.add_argument(
        "-C",
        "--colorSelection",
        type=str.lower,
        choices=["nearest", "fixed"],
        default="nearest",
        help="The color selection method used: [nearest, fixed] (default: nearest).",
    )
    parser.add_argument(
        "-a",
        "--autoColor",
        nargs="?",
        type=int,
        const=16,
        help="Size of sampled color palette from the most prominent colors in the picture (defalut: 16).",
    )
    parser.add_argument(
        "-i",
        "--invert",
        action="store_false",
        help="Invert the output of the image (default: light characters on black background).",
    )

    ### ASCII SAMPLING ARGS ###

    parser.add_argument(
        "-n",
        "--cols",
        type=int,
        default=120,
        help="The number of characters on the width of the output image (default: 120).",
    )
    parser.add_argument(
        "-l",
        "--scale",
        type=float,
        default=0.6,
        help="The width-to-height ratio of the pixels sampled for each character (default: 0.6).",
    )
    parser.add_argument(
        "-t",
        "--chars",
        default=ascii_chars,
        help="The ASCII characters to be used or select from presets: [printable, alphanumeric, alpha, numeric, lower, upper, tech, symbols] (default: printable)",
    )
    parser.add_argument(
        "-f",
        "--constrastFactor",
        type=float,
        default=1.3,
        help="Contrast factor: <1 less contrast, 1 no change, >1 more contrast (default: 1.3).",
    )

    parser.add_argument(
        "-T",
        "--sampling",
        type=str.lower,
        choices=["resize", "median", "mean"],
        default="resize",
        help="The sampling method used: [resize, median, mean] (default: resize).",
    )
    parser.add_argument(
        "-F",
        "--fontPath",
        required=False,
        help="The path to the font to be used (default: SFMono-Medium).",
    )

    ### OUTPUT ARGS ###

    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=1920,
        help="The resolution of the output image (default: 1920)",
    )
    parser.add_argument(
        "-S",
        "--save",
        nargs="?",
        const="",
        help="Save ASCII image as inputed path (default: '../out/<filename>_ascii.png').",
    )
    parser.add_argument(
        "-O", "--out", dest="outFile", required=False, help="Output text location."
    )
    parser.add_argument(
        "-P",
        "--print",
        action="store_true",
        help="Print ASCII text to output (default: false).",
    )
    parser.add_argument(
        "-H",
        "--hide",
        action="store_true",
        help="Do not open image after conversion (default: false).",
    )

    # --------------  PARSING ARGUMENTS -------------- #
    args = parser.parse_args()

    # open image and convert to RGB
    filename = Path(args.filename)
    if filename.exists():
        image = Image.open(args.filename).convert("RGB")
        image = filterImage(image, args.constrastFactor)
    else:
        print("ERROR: Image does not exist.")
        exit(0)

    # set text output file
    outFile = "out.txt"
    if args.outFile:
        outFile = args.outFile

    # set img output file
    saveOutput = False
    if args.save == "":
        imgOutFile = Path("../out/{}_ascii.png".format(filename.stem))
        saveOutput = True
    elif args.save:
        imgOutFile = Path(args.save)
        saveOutput = True

    # verify font path
    fontPath = args.fontPath
    if fontPath:
        fontFile = Path(fontPath)
        if not fontFile.exists():
            print("ERROR: Font file does not exist.")
            exit(0)

    # sort chars by char density
    if args.chars == "printable":
        chars = ascii_chars
    elif args.chars == "alphanumeric":
        chars = ascii_alphanumeric
    elif args.chars == "alpha":
        chars = ascii_alpha
    elif args.chars == "lower":
        chars = ascii_lowercase
    elif args.chars == "upper":
        chars = ascii_uppercase
    elif args.chars == "tech":
        chars = ascii_tech
    elif args.chars == "numeric":
        chars = ascii_numeric
    elif args.chars == "symbols":
        chars = ascii_symbols
    elif args.chars == "std1":
        chars = ascii_std_1
    elif args.chars == "std2":
        chars = ascii_std_2
    else:
        chars = args.chars
    orderedChars = orderChars(chars, font_path=fontPath)

    # set output size
    outputSize = autoSize(image, resolution=args.resolution)

    # set sampling scheme
    if args.sampling == "median":
        sampling = BLOCK_REDUCE
    elif args.sampling == "mean":
        sampling = DOWNSCALE_MEAN_REDUCE
    else:
        sampling = OPENCV_RESIZE

    # set sampling scheme
    if args.colorSelection == "fixed":
        colorSelection = FIXED
    else:
        colorSelection = NEAREST

    # set color scheme
    if args.autoColor:
        num_of_colors = args.autoColor
        colors = autoColor(image, num_of_colors, greyscale=args.greyscale)
    else:
        colors = defaultPalatte(args.samples, args.colorPalatte)

    # --------------  GENERATE ASCII ART -------------- #

    print("Generating ASCII art...")
    start = time.perf_counter()

    # convert image to ascii txt
    aimg, cimg = covertImageToAscii(
        image,
        colors,
        args.cols,
        args.scale,
        orderedChars,
        args.invert,
        sampling=sampling,
        colorSelection=colorSelection,
    )

    # make image from text
    image = textToImage(aimg, cimg, args.invert, outputSize, font_path=fontPath)

    # write to text file
    f = open(outFile, "w")
    background_color = "\033[40m" if args.invert else "\033[107m"
    for line, line_colors in zip(aimg, cimg):
        f.write(background_color)
        for c, color in zip(line, line_colors):
            ansi_color = RGB2ANSI256(*color)
            f.write(ansi_color + c)
        f.write("\033[0m\n")
    f.close()

    # print output
    if args.print:
        f = open(outFile, "r")
        print(f.read(), end="")
        f.close()

    # save/show the image
    if saveOutput:
        # Confirm if about to overwrite a file
        if imgOutFile.is_file():
            response = input(
                "Warning: file '{}' already exists.\nContinue and overwrite this file? (y/n) ".format(
                    imgOutFile
                )
            )
            if response.upper() != "Y":
                exit(0)

        imgOutFile.parent.mkdir(parents=True, exist_ok=True)
        image.save(imgOutFile)

        # Show image
        if not args.hide:
            if sys.platform == "win32":
                os.startfile(imgOutFile)
            else:
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.call([opener, imgOutFile])
    elif not args.hide:
        image.show()

    end = time.perf_counter()
    print(f"Completed {end - start:0.4f} seconds")


# call main
if __name__ == "__main__":
    main()
