import sys, argparse, time
import numpy as np
import matplotlib.pyplot as plt
import os, sys, subprocess
import string

from PIL import Image, ImageFont, ImageDraw

ascii_chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ "
ascii_letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
ascii_lowercase = "gdbqpmhwkaeyounjslfictzxrv"
ascii_uppercase = "QBMDRWGNOHSUKEPCAZXVJFIYTL"
ascii_upper_lower = "QBMDRWGNOHSUKEPCAZXVJFIYTLdbqpmhwkaeyounjslfictzxrv"
gscale1 = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "


# 10 levels of gray
gscale2 = "@%#*+=-:. "


def getCharDensity(char):
    font_path = "fonts/SFMono-Semibold.otf"
    font_size = 200
    try:
        font = ImageFont.truetype(font_path, size=font_size)
    except IOError:
        font = ImageFont.load_default()
        print("Could not use chosen font. Using default.")

    # init image
    char_w = font.getsize(char)[0]
    char_h = round(4 / 3 * font_size)
    # print((char_w, char_h))
    image = Image.new("1", (char_w, char_h), color=0)

    # draw text to image
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), char, fill=1, font=font, spacing=0)

    # convert image to np array
    im = np.array(image)

    # Count white pixels
    white = np.count_nonzero(im == True)
    # print(f"white: {white}")

    # Count black pixels
    black = np.size(im) - white
    # print(f"black: {black}")

    # calc ratio
    density = white / black
    # print(f"ratio: {ratio}")
    return density


def orderChars(chars):
    char_dict = []
    for c in chars:
        density = getCharDensity(c)
        char_dict.append((c, density))

    char_dict.sort(key=lambda x: x[1], reverse=True)

    ordered_chars = [x[0] for x in char_dict]
    ordered_chars = "".join(ordered_chars)

    print(ordered_chars)


def main():
    # orderChars(ascii_chars)

    print(string.ascii_letters)
    orderChars(string.ascii_uppercase)
    # orderChars(gscale1)
    # print(gscale1)
    # orderChars(gscale2)
    # print(gscale2)


# call main
if __name__ == "__main__":
    main()
