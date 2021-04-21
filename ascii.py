# Python code to convert an image to ASCII image.
import sys, argparse, time
import numpy as np
from scipy.ndimage.filters import gaussian_filter, percentile_filter
from scipy import ndimage
import math
import matplotlib.pyplot as plt
  
from PIL import Image, ImageFilter, ImageFont, ImageDraw, ImageOps
  
# gray scale level values from: 
# http://paulbourke.net/dataformats/asciiart/
  
# 70 levels of gray
gscale1 = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
  
# 10 levels of gray
gscale2 = '@%#*+=-:. '

# color codes
# colors = [ '\033[0m']
# colors = [ '\033[37m', '\033[33m', '\033[31m', '\033[34m', '\033[0m']     # white, yellow, red, blue
# colors = [ '\033[37m', '\033[94m', '\033[35m', '\033[34m', '\033[0m']   # cool - white, light blue, magenta, blue

colors = [ '\033[97m', '\033[37m', '\033[90m', '\033[0m'] # 16bit grayscale
# colors = [ '\033[38;5;255m', '\033[38;5;248m', '\033[38;5;241m', '\033[0m']   # grayscale 1
# colors = [ '\033[38;5;255m', '\033[38;5;250m', '\033[38;5;244m', '\033[38;5;239m', '\033[38;5;235m', '\033[0m' ] # grayscale 2

def normalizeImage():
    """
    Given PIL Image, return average value of grayscale value
    """
    # get image as numpy array
    im = np.array(image)
  
    # get max and min values
    min_val = np.amin(im.reshape(w*h))
    max_val = np.amax(im.reshape(w*h))
    
    return min_val, max_val
    
def filterImage(image):
    return image.filter(ImageFilter.EDGE_ENHANCE_MORE)
  
def covertImageToAscii(fileName, cols, scale, moreLevels, invertImg):
    
    """
    Given Image and dims (rows, cols) returns an m*n list of Images 
    """
    # declare globals
    global gscale1, gscale2
  
    # open image and convert to grayscale
    image = Image.open(fileName).convert('L')
    
  
    # store dimensions
    W, H = image.size[0], image.size[1]
    print("input image dims: %d x %d" % (W, H))
  
    # compute width of tile
    w = W / cols
  
    # compute tile height based on aspect ratio and scale
    h = w / scale
  
    # compute number of rows
    rows = int(H/h)
      
    print("cols: %d, rows: %d" % (cols, rows))
    print("tile dims: %d x %d" % (w, h))
  
    # check if image size is too small
    if cols > W or rows > H:
        print("Image too small for specified cols!")
        exit(0)
    
    # get image as numpy array
    im = np.array(image)
    
    # ------ todo: remove ------
    # f = plt.figure()
    # f.add_subplot(1,2, 1)
    # plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    
    # im = ndimage.median_filter(im, 3)
    
    # f.add_subplot(1, 2, 2)
    # plt.imshow(block_mean, cmap='gray', vmin=0, vmax=255)
    # plt.show(block=True)
    # ------ todo: remove ------
    

    im = im.transpose()
    print("image array " + str(im.shape))
    
    # apply filter
    if (invertImg):
        im = 255 - im

    
    """
    Given numpy array of a b/w image, get the average value tile array
    """
    avgs = []
    
    for j in range(rows):
        row_avgs = []
        y1 = int(j*h)
        y2 = int((j+1)*h)
        
        # correct last tile
        if j == rows-1:
            y2 = H
        
        for i in range(cols):
            x1 = int(i*w)
            x2 = int((i+1)*w)
            
            # correct last tile
            if i == cols-1:
                x2 = W
            
            # get avg brightness array of tile
            avg = np.median(im[x1:x2,y1:y2])
            row_avgs.append(avg)
        
        avgs.append(row_avgs)  
    
    tiles = np.array(avgs)
    
    # ------ todo: remove ------
    # f = plt.figure()
    # f.add_subplot(1,2, 1)
    # plt.imshow(tiles, cmap='gray', vmin=0, vmax=255)
    
    # # filtered_tiles = ndimage.median_filter(tiles, 3)
    
    # f.add_subplot(1,2, 2)
    # plt.imshow(filtered_tiles, cmap='gray', vmin=0, vmax=255)
    # plt.show(block=True)
    # tiles = filtered_tiles
    # ------ todo: remove ------
    
    
    
    
    
    # tiles = gaussian_filter(tiles, sigma=1.5)
    # tiles = percentile_filter(tiles, percentile=20, size=2)
    print(tiles.min(), tiles.max())
    
    # normalize tile value array
    min_val = np.percentile(tiles, 0)
    max_val = np.percentile(tiles, 80)
    tiles = np.clip(tiles, min_val, max_val)
    
    print(tiles.min(), tiles.max())

    # min_val = 0
    # max_val = 255
    normal_tiles = (tiles - min_val) / (max_val - min_val)
    
    print(normal_tiles.min(), normal_tiles.max())
  
    # ascii image is a list of character strings
    aimg = []
    # generate list of dimensions
    for j in range(rows):
        # append an empty string
        aimg.append("")
        
        for i in range(cols):
            # todo: add command
            # color_index = int((len(colors)-1) * normal_tiles[j][i])
            # gsval = colors[color_index]
            
            # look up ascii char
            # min_val = color_index / (len(colors) - 1)
            # max_val = (color_index + 1) / (len(colors) - 1)
            # char_ratio = (normal_tiles[j][i] - min_val) / (max_val - min_val)
            # # if (normal_tiles[j][i] == 1):
            # #     char_ratio = 1
            # print("norm: %.2f, char_ratio: %.2f, color_index: %.2f, min_val: %.2f, max_val: %.2f" % (normal_tiles[j][i], char_ratio, color_index, min_val, max_val))
            gsval = ""
            char_ratio = normal_tiles[j][i]
            
            if moreLevels:
                gsval += gscale1[int(69 * char_ratio)]
            else:
                gsval += gscale2[int(9 * char_ratio)]
                
            # gsval += '\033[0m'
            
  
            # append ascii char to string
            aimg[j] += gsval
      
    # return txt image
    return aimg


# --------- TODO: refactor -------------- #

PIXEL_ON = 0  # PIL color to use for "on"
PIXEL_OFF = 255  # PIL color to use for "off"

def text_image(text_path, inverted, font_path=None):
    """Convert text file to a grayscale image with black characters on a white background.

    arguments:
    text_path - the content of this file will be converted to an image
    font_path - path to a font file (for example impact.ttf)
    """
    grayscale = 'L'
    # parse the file into lines
    with open(text_path) as text_file:  # can throw FileNotFoundError
        lines = tuple(l.rstrip() for l in text_file.readlines())

    # choose a font (you can see more detail in my library on github)
    large_font = 20  # get better resolution with larger size
    font_path = font_path or 'fonts/SFMono-Semibold.otf'  # Courier New. works in windows. linux may need more explicit path
    try:
        font = ImageFont.truetype(font_path, size=large_font)
    except IOError:
        font = ImageFont.load_default()
        print('Could not use chosen font. Using default.')

    # make the background image based on the combination of font and lines
    pt2px = lambda pt: int(round(pt * 96.0 / 72))   # function that converts points to pixels
    max_width_line = max(lines, key=lambda s: font.getsize(s)[0])   # get line with largest width
    # max height is adjusted down because it's too large visually for spacing
    test_string = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    max_height = pt2px(font.getsize(test_string)[1])
    max_width = pt2px(font.getsize(max_width_line)[0])
    char_width = round(max_width / len(max_width_line))
    height = max_height * len(lines)  # perfect or a little oversized
    width = int(max_width + 4)  # a little oversized
    image = Image.new(grayscale, (width, height), color=PIXEL_OFF)
    draw = ImageDraw.Draw(image)

    # draw each line of text
    vertical_position = 2
    horizontal_position = 4
    line_spacing = int(round(max_height * 0.85))  # reduced spacing seems better
    # line_spacing = max_height
    char_spacing = round(char_width * 0.75)
    for line in lines:
        hor_pos = horizontal_position
        for c in line:
            draw.text((hor_pos, vertical_position),
                    c, fill=PIXEL_ON, font=font)
            hor_pos += char_spacing 
        # draw.text((horizontal_position, vertical_position), line, fill=PIXEL_ON, font=font)
        vertical_position += line_spacing
    # crop the text
    c_box = ImageOps.invert(image).getbbox()
    # c_box = image.getbbox()
    image = image.crop(c_box)
    image = ImageOps.invert(image)
    return image

# ---------------------------------------- #

# main() function
def main():
    # create parser
    descStr = "This program converts an image into ASCII art."
    parser = argparse.ArgumentParser(description=descStr)
    # add expected arguments
    parser.add_argument('--file', dest='imgFile', required=True)
    parser.add_argument('--scale', dest='scale', required=False)
    parser.add_argument('--out', dest='outFile', required=False)
    parser.add_argument('--cols', dest='cols', required=False)
    parser.add_argument('--morelevels',dest='moreLevels',action='store_true')
    parser.add_argument('--invert',dest='invert',action='store_true')
  
    # parse args
    args = parser.parse_args()
    
    imgFile = args.imgFile
  
    # set output file
    outFile = 'out.txt'
    if args.outFile:
        outFile = args.outFile
  
    # set scale default as 0.43 which suits
    # a Courier font
    scale = 0.5
    if args.scale:
        scale = float(args.scale)
  
    # set cols
    cols = 80
    if args.cols:
        cols = int(args.cols)
  
    print('generating ASCII art...')
    
    start = time.perf_counter()
    # convert image to ascii txt
    aimg = covertImageToAscii(imgFile, cols, scale, args.moreLevels, args.invert)
    end = time.perf_counter()
    print(f"Completed {end - start:0.4f} seconds")
  
    # open file
    f = open(outFile, 'w')
  
    # write to file
    for row in aimg:
        print(row)
        f.write(row + '\n')
  
    # cleanup
    f.close()
    print("ASCII art written to %s" % outFile)
    
    start = time.perf_counter()
    image = text_image(outFile, args.invert)
    image.show()
    end = time.perf_counter()
    print(f"Completed {end - start:0.4f} seconds")
    # image.save('content.png')
  
# call main
if __name__ == '__main__':
    main()
