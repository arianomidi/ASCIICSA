"""
A program that orders characters based on their density.

Author: Arian Omidi
Email: arian.omidi@icloud.com
GitHub: https://github.com/ArianOmidi
Date: 2021-06-01
"""
from PIL import Image, ImageFont, ImageDraw
import numpy as np

# define standard character sets
ascii_chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
ascii_alphanumeric = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
ascii_alpha = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
ascii_lowercase = "abcdefghijklmnopqrstuvwxyz"
ascii_uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ascii_numeric = "0123456789"
ascii_tech = "0123456789!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
ascii_symbols = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
ascii_std_1 = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'."
ascii_std_2 = "@%#*+=-:."


def getCharDensity(char, font, font_size):
    """
    Return the character density of a given character.

    arguments:
    char - character to determine its density
    font - the font to be used (default: SFMono-Medium)
    """

    # init image
    char_w = font.getsize(char)[0]
    char_h = font_size
    image = Image.new("1", (char_w, char_h), color=0)

    # draw text to image
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), char, fill=1, font=font, spacing=0)

    # convert image to np array
    im = np.array(image)

    # Count white and black pixels
    white = np.count_nonzero(im == True)
    black = np.size(im) - white

    # return the ratio of used pixels
    return white / black


def orderChars(chars, font_path=None):
    """
    Order the characters in a given string by their density.

    arguments:
    chars - string of characters to order
    font_path - path to the font to be used (default: SFMono-Medium)
    """

    font_path = font_path or "../fonts/SFMono-Medium.otf"
    try:
        font = ImageFont.truetype(font_path, size=200)
        font_size = font.getsize(ascii_chars)[1]
    except IOError:
        font = ImageFont.load_default()
        font_size = font.getsize(ascii_chars)[1]
        print("Warning: Could not use chosen font. Using default.")

    char_dict = []
    for c in chars:
        density = getCharDensity(c, font, font_size)
        char_dict.append((c, density))

    char_dict.sort(key=lambda x: x[1], reverse=True)

    ordered_chars = [x[0] for x in char_dict]
    ordered_chars = "".join(ordered_chars)
    ordered_chars += " "

    return ordered_chars
