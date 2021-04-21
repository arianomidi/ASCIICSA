# Python code to convert an image to ASCII image.
import sys, argparse, time
import numpy as np
from scipy.ndimage.filters import gaussian_filter, percentile_filter
from scipy import ndimage
import math
import matplotlib.pyplot as plt
  
from PIL import Image, ImageFilter
  
# gray scale level values from: 
# http://paulbourke.net/dataformats/asciiart/
  
# 70 levels of gray
gscale1 = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
  
# 10 levels of gray
gscale2 = '@%#*+=-:. '

# color codes
# colors = [ '\033[38;5;255m', '\033[38;5;248m', '\033[38;5;241m' ]
colors = [ '\033[38;5;255m', '\033[38;5;250m', '\033[38;5;244m', '\033[38;5;238m', '\033[38;5;232m']

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
    max_val = np.percentile(tiles, 100)
    tiles = np.clip(tiles, min_val, max_val)

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
            gsval = colors[round((len(colors)-1) * normal_tiles[j][i])]
            
            # look up ascii char
            if moreLevels:
                gsval += gscale1[int(69 * normal_tiles[j][i])]
            else:
                gsval += gscale2[int(9 * normal_tiles[j][i])]
                
            gsval += '\033[0m'
  
            # append ascii char to string
            aimg[j] += gsval
      
    # return txt image
    return aimg

def prRed(skk): print("\033[91m{}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m{}\033[00m" .format(skk))
def prYellow(skk): print("\033[93m{}\033[00m" .format(skk))
def prLightPurple(skk): print("\033[94m{}\033[00m" .format(skk))
def prPurple(skk): print("\033[95m{}\033[00m" .format(skk))
def prCyan(skk): print("\033[96m{}\033[00m" .format(skk))
def prLightGray(skk): print("\033[97m{}\033[00m" .format(skk))
def prBlack(skk): print("\033[98m{}\033[00m" .format(skk))


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
  
# call main
if __name__ == '__main__':
    main()
