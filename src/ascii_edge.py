from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2
import string

img_path = "./data/zebra.png"
kernel_size = 5

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


def detectEdges():
    img = cv2.imread(img_path, 0)

    blur_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    sobel_img = cv2.Sobel(
        src=blur_img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3
    )  # Combined X and Y Sobel Edge Detection

    cv2.imshow("OpenCV Image Reading", sobel_img)
    cv2.waitKey(0)


detectEdges()
