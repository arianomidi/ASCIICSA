"""
A program that converts an image into animated ASCII art v2.0

Author: Arian Omidi
Email: arian.omidi@icloud.com
GitHub: https://github.com/ArianOmidi
Date: 2021-06-01
"""
import os, sys, subprocess
from pathlib import Path
from shutil import rmtree

import cv2
from PIL import Image
import numpy as np

from tqdm import tqdm
import argparse

from ascii import convertImageToAscii, autoColor, autoSize, defaultPalatte
from ascii_video import *
from color import *
from char_density import *


# ======================  ASCII CONVERSION METHODS  ====================== #


def animateAsciiImage(
    image,
    out,
    fps,
    gs_inc,
    colors,
    cols,
    scale,
    chars,
    invert,
    size=None,
):
    """
    Converts given image to animated ASCII Art.

    arguments:
    image - PIL image
    out - save location of the ASCII animation
    fps - frames per sec
    gs_inc - grayscale increment
    colors - the color pallate to be used as an RGB array
    cols - the width of the output image in number of letters
    scale - the ratio of rows to cols
    chars - the characters used in the ASCII image
    invert - if the backgroung should be black or white
    size - size of the outputed video
    """

    # init output video
    video = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    image = image.resize(size)

    # init image arrays
    image = image.convert("L")
    frame_arr = np.array(image).astype(np.int32)

    # init mask
    max_val = 255
    min_val = 0
    mask = (min_val <= frame_arr) & (frame_arr <= max_val)
    inc_mask = frame_arr <= max_val

    # get total frames
    total_frames = round(2 * (max_val - min_val) / gs_inc)

    # init progress bar
    with tqdm(range(total_frames), ncols=75, unit="f") as pbar:
        for frame_num in pbar:
            # update image according to mask
            np.putmask(frame_arr, mask & inc_mask, frame_arr + gs_inc)
            np.putmask(frame_arr, mask & ~inc_mask, frame_arr - gs_inc)
            # update inc mask
            inc_mask = np.logical_or(frame_arr <= min_val, inc_mask)
            inc_mask = np.logical_and(frame_arr < max_val, inc_mask)

            # clip the image to vals from 0 to 255
            frame_arr = np.clip(frame_arr, 0, 255)

            # convert OpenCV image to PIL and to RGB
            frame = Image.fromarray(frame_arr).convert("RGB")

            # convert image to ascii
            ascii_img = convertImageToAscii(
                frame,
                colors,
                cols,
                scale,
                chars,
                invert,
                size=size,
            )

            # convert ascii frame from PIL to OpenCV and add to the video
            ascii_img_cv = convertPIL2OpenCV(ascii_img)
            video.write(ascii_img_cv)

            frame_num += 1

    # Release all space and windows once done
    video.release()
    cv2.destroyAllWindows()


# ======================  MAIN PROGRAM  ====================== #


def main():
    # create parser
    descStr = "This program converts an image into moving ASCII art."
    parser = argparse.ArgumentParser(description=descStr)
    # add expected arguments
    parser.add_argument("filename")

    parser.add_argument("-g", "--greyscale", type=int, default=8)
    parser.add_argument("-n", "--cols", type=int, default=120)
    parser.add_argument("-l", "--scale", type=float, default=0.6)
    parser.add_argument("-t", "--chars", default=ascii_chars)
    parser.add_argument("-i", "--invert", action="store_false")
    parser.add_argument("-f", "--fps", type=int, default=16)
    parser.add_argument("-d", "--inc", type=int, default=2)
    parser.add_argument("-r", "--resolution", type=int, default=1920)

    parser.add_argument("-O", "--out", dest="outFile", required=False)
    parser.add_argument("-H", "--hide", action="store_true")

    # --------------  PARSING ARGUMENTS  -------------- #
    # parse args
    args = parser.parse_args()

    # open image and convert to RGB
    filename = Path(args.filename)
    if filename.exists():
        image = Image.open(args.filename).convert("RGB")
    else:
        print("ERROR: Image does not exist.")
        exit(0)

    # set text output file
    outFile = Path("../out/animated/{}_animated_ascii.mp4".format(filename.stem))
    if args.outFile:
        outFile = Path(args.outFile)

    # set output size
    outputSize = autoSize(image, resolution=args.resolution)

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
    orderedChars = orderChars(chars)

    # set color scheme
    colors = defaultPalatte(args.greyscale)

    # --------------  GENERATE ASCII ART -------------- #
    # Confirm if about to overwrite a file
    if outFile.is_file():
        response = input(
            "Warning: file '{}' already exists.\nContinue and overwrite this file? (y/n) ".format(
                outFile
            )
        )
        if response.upper() != "Y":
            exit(0)

    outFile.parent.mkdir(parents=True, exist_ok=True)

    print("Generating ASCII art...")

    # Convert image to moving ascii image
    animateAsciiImage(
        image,
        outFile,
        args.fps,
        args.inc,
        colors,
        args.cols,
        args.scale,
        orderedChars,
        args.invert,
        size=outputSize,
    )

    # Show video
    if not args.hide:
        if sys.platform == "win32":
            os.startfile(outFile)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, outFile])


# call main
if __name__ == "__main__":
    main()
