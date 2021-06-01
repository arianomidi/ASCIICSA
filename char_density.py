from PIL import Image, ImageFont, ImageDraw
import numpy as np

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


def getCharDensity(char, font_path=None):
    font_path = font_path or "fonts/SFMono-Medium.otf"
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


def orderChars(chars, font_path=None):
    char_dict = []
    for c in chars:
        density = getCharDensity(c, font_path)
        char_dict.append((c, density))

    char_dict.sort(key=lambda x: x[1], reverse=True)

    ordered_chars = [x[0] for x in char_dict]
    ordered_chars = "".join(ordered_chars)
    ordered_chars += " "

    return ordered_chars
