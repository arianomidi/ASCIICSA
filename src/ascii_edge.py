from PIL import Image, ImageFont, ImageDraw
import numpy as np
import string

chars = " .,<>[]()\\!|/[]"
charsTest = "?{|"


def getCharImages(chars, font_path=None):
    font_path = font_path or "./fonts/SFMono-Medium.otf"
    try:
        font = ImageFont.truetype(font_path, size=200)
        font_size = font.getsize(string.printable)[1]
    except IOError:
        font = ImageFont.load_default()
        font_size = font.getsize(string.printable)[1]
        print("Warning: Could not use chosen font. Using default.")

    charImgs = []
    for char in chars:
        # init image
        char_w = font.getsize(char)[0]
        char_h = font_size
        image = Image.new("L", (char_w, char_h), color=0)

        # draw text to image
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), char, fill=1, font=font, spacing=0)

        # convert image to np array
        charImgs.append(np.array(image))

    return np.array(charImgs)


print(getCharImages(charsTest))
