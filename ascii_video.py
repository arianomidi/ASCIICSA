"""
A program that converts a video into ASCII art v2.0

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

from ascii import convertImageToAscii, autoColor
from color import *
from char_density import *


# ======================  HELPER METHODS  ====================== #


def convertOpenCV2PIL(img_cv):
    img_pil = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_pil)


def convertPIL2OpenCV(img_pil):
    img_arr = np.array(img_pil)
    return cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)


def extractFrame(filename, out=None, loc=0.5):
    """
    Extracts a frame from video.

    arguments:
    filename - name of source video
    out - save location of the frame
    loc - location of frame: 0 is the first frame and 1 is the last (default: 0.5)
    """
    cam = cv2.VideoCapture(filename)

    ret, frame = cam.read()
    while cam.get(cv2.CAP_PROP_POS_AVI_RATIO) <= loc:
        ret = cam.grab()
        if ret:
            ret, frame = cam.retrieve()
        else:
            break

    if out:
        cv2.imwrite(out, frame)

    cam.release()
    cv2.destroyAllWindows()

    return frame


# ======================  VIDEO MANIPULATION  ====================== #


def boomerang(filename, out=None, fps=24, progress_bar=True):
    """
    Converts video into a Boomerang.

    arguments:
    filename - name of source video
    out - save location of the boomerang
    fps - frames per sec (default: 24)
    progress_bar - display a progress bar if True
    """
    file = Path(str(filename))
    # Check if file given is valid
    if not file.exists():
        print("ERROR: Video does not exist.")
        exit(0)

    frames_path = file.parent / "frames"
    frames_path.mkdir(parents=True, exist_ok=True)

    # create the frames
    vid_to_img(file, frames_path, progress_bar=progress_bar)

    # get frames
    frames = sorted(frames_path.glob("*"))
    # create boomerang effect by looping the frames
    frames.extend(reversed(frames))

    # save frames to video
    out = out or Path("{}/{}_boomerang.mp4".format(file.parent, file.stem))
    img_to_vid(out, fps, frames=frames, progress_bar=progress_bar)

    # delete frame dir
    try:
        rmtree(frames_path)
    except OSError as e:
        print("Error: %s : %s" % (frames_path, e.strerror))


def img_to_vid(out, fps, in_path=None, frames=None, progress_bar=False):
    """
    Converts frames into a video.

    arguments:
    out - save location of the video
    fps - frames per sec
    in_path - directory of frames (optional)
    frames - list of frames (optional)
    progress_bar - display a progress bar if True
    """
    # Get all the frame paths
    if in_path:
        frames = sorted(in_path.glob("*"))
    elif not frames:
        print("ERROR: 'img_to_vid' - input path or frames not given.")
        exit(0)

    # Get the dimensions of the video
    frame = cv2.imread(str(frames[0]))
    height, width, _ = frame.shape

    video = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    # create progress bar
    with tqdm(
        frames,
        desc="Creating boomerang",
        ncols=75,
        unit="f",
        disable=(not progress_bar),
    ) as pbar:
        for frame in pbar:
            video.write(cv2.imread(str(frame)))

    cv2.destroyAllWindows()
    video.release()


def vid_to_img(filename, out, progress_bar=False):
    """
    Converts a video into frames.

    arguments:
    filename - name of source video
    out - save location of the frames
    progress_bar - display a progress bar if True
    """
    # init video capture and frame number
    cam = cv2.VideoCapture(str(filename))
    total_frames = round(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    currentframe = 0

    with tqdm(
        total=total_frames,
        desc="Converting video to frames",
        ncols=75,
        unit="f",
        disable=(not progress_bar),
    ) as pbar:
        while True:
            # reading from frame
            ret, frame = cam.read()

            if ret:
                num = str(currentframe).zfill(6)
                # if video is still left continue creating images
                name = out / "frame{}.jpg".format(num)

                # writing the extracted images
                cv2.imwrite(str(name), frame)

                # increasing counter so that it will
                # show how many frames are created
                currentframe += 1

                # update the progress bar
                pbar.update(1)
            else:
                break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


# ======================  ASCII CONVERSION METHODS  ====================== #


def convertVideoToAscii(
    filename,
    out,
    fps,
    colors,
    frameAutoColor,
    colorSampleRate,
    startCols,
    endCols,
    scale,
    chars,
    invert,
    size=None,
):
    """
    Converts given video into ASCII art.

    arguments:
    filename - name of the input video
    out - save location of the frames
    fps - frames per sec
    colors - the color pallate to be used as an RGB array
    frameAutoColor - sample colors at colorSampleRate if True
    colorSampleRate - the rate at which to auto sample colors
    startCols - the width of the output image in number of letters on the first frame
    endCols - the width of the output image in number of letters on the last frame
    scale - the ratio of rows to cols
    chars - the characters used in the ASCII image
    invert - if the backgroung should be black or white
    size - size of the outputed video
    """
    # init video capture
    cam = cv2.VideoCapture(str(filename))
    total_frames = round(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    # init output video
    if not size:
        success, frame = cam.read()
        frame_pil = convertOpenCV2PIL(frame).convert("RGB")
        ascii_img = convertImageToAscii(
            frame_pil, colors, startCols, scale, chars, invert, size=size
        )
        size = ascii_img.size

    video = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    # init progress bar
    frame_num = 0
    with tqdm(total=total_frames, ncols=75, unit="f") as pbar:
        while True:
            # get next frame if available
            success, frame = cam.read()
            if not success:
                break

            # update the progress bar
            pbar.update(1)

            # convert OpenCV image to PIL and to greyscale
            frame = convertOpenCV2PIL(frame).convert("RGB")

            # auto color every n frames
            if frameAutoColor and frame_num % colorSampleRate == 0:
                colors = frameAutoColor(frame)

            # get number of cols
            video_pos = frame_num / total_frames
            cols = round((endCols - startCols) * video_pos + startCols)

            # convert image to ascii
            ascii_img = convertImageToAscii(
                frame, colors, cols, scale, chars, invert, size=size
            )

            # convert ascii frame from PIL to OpenCV and add to the video
            ascii_img_cv = convertPIL2OpenCV(ascii_img)
            video.write(ascii_img_cv)

            frame_num += 1

    # Release all space and windows once done
    cam.release()
    video.release()
    cv2.destroyAllWindows()


# ======================  MAIN PROGRAM  ====================== #


def main():
    # create parser
    descStr = "This program converts a video into ASCII art."
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
        "-c", "--color", dest="colorScheme", type=int, nargs="?", const=16
    )
    parser.add_argument("-a", "--autoColor", action="store_true")
    parser.add_argument("-R", "--colorSampleRate", type=int, nargs="?", const=-1)
    parser.add_argument("-n", "--cols", type=int, nargs="+", default=120)
    parser.add_argument("-l", "--scale", type=float, default=0.6)
    parser.add_argument("-t", "--chars", default=ascii_chars)
    parser.add_argument("-i", "--invert", action="store_false")
    parser.add_argument("-f", "--fps", type=int, required=False)
    parser.add_argument("-r", "--resolution", type=int, default=1920)

    parser.add_argument("-O", "--out", dest="outFile", required=False)
    parser.add_argument("-H", "--hide", action="store_true")
    parser.add_argument("-T", "--test", action="store_true")

    # --------------  PARSING ARGUMENTS -------------- #
    # parse args
    args = parser.parse_args()

    # open image and convert to grayscale
    filename = Path(args.filename)
    # Check if file given is valid
    if not filename.exists():
        print("ERROR: Video does not exist.")
        exit(0)

    # set text output file
    outFile = Path("out/video/{}_ascii.mp4".format(filename.stem))
    if args.outFile:
        outFile = Path(args.outFile)

    # set scale default as 0.6 which suits
    # a Courier font
    scale = args.scale

    # set cols
    startCols = 120
    endCols = 120
    if args.cols:
        startCols = int(args.cols[0])
        if len(args.cols) > 1:
            endCols = int(args.cols[1])
        else:
            endCols = startCols

    # set fps
    fps = args.fps
    if not args.fps:
        cam = cv2.VideoCapture(str(filename))
        fps = round(cam.get(cv2.CAP_PROP_FPS))
        cam.release()
        cv2.destroyAllWindows()

    # set output size
    resolution = args.resolution

    cam = cv2.VideoCapture(str(filename))
    width = int(cam.get(3))
    height = int(cam.get(4))
    cam.release()
    cv2.destroyAllWindows()

    if width >= height:
        size = (resolution, round((resolution / width) * height))
    else:
        size = (round((resolution / height) * width), resolution)

    # set sample rate
    sampleRate = fps
    if args.colorSampleRate:
        sampleRate = args.colorSampleRate

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
    frameAutoColor = None
    if args.autoColor:
        num_of_colors = args.colorScheme or args.greyscaleScheme
        is_greyscale = args.colorScheme == None
        frameAutoColor = lambda frame: autoColor(
            frame, num_of_colors, greyscale=is_greyscale
        )

        sample_frame = convertOpenCV2PIL(extractFrame(str(filename))).convert("RGB")
        colors = frameAutoColor(sample_frame)
    else:
        if args.colorScheme:
            colors = ansi16_rgb()
        else:
            colors = []
            for i in range(args.greyscaleScheme):
                color = int(i * 255 / (args.greyscaleScheme - 1))
                colors.append((color, color, color))

    # --------------  TEST FRAMES (OPTIONAL) -------------- #
    if args.test:
        # get frames
        start_frame = convertOpenCV2PIL(extractFrame(str(filename), loc=0)).convert(
            "RGB"
        )
        mid_frame = convertOpenCV2PIL(extractFrame(str(filename), loc=0.5)).convert(
            "RGB"
        )
        end_frame = convertOpenCV2PIL(extractFrame(str(filename), loc=1)).convert("RGB")

        # set color
        if frameAutoColor:
            start_colors = frameAutoColor(start_frame)
            mid_colors = frameAutoColor(mid_frame)
            end_colors = frameAutoColor(end_frame)
        else:
            start_colors = mid_colors = end_colors = colors

        start_ascii = convertImageToAscii(
            start_frame,
            start_colors,
            startCols,
            scale,
            orderedChars,
            args.invert,
            size=size,
        )
        mid_ascii = convertImageToAscii(
            mid_frame,
            mid_colors,
            (startCols + endCols) // 2,
            scale,
            orderedChars,
            args.invert,
            size=size,
        )
        end_ascii = convertImageToAscii(
            end_frame,
            end_colors,
            endCols,
            scale,
            orderedChars,
            args.invert,
            size=size,
        )

        start_ascii.show()
        mid_ascii.show()
        end_ascii.show()

        response = input("Continue? (y/n) ".format(outFile))
        if response.upper() != "Y":
            exit(0)

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

    # --------------  GENERATE ASCII ART -------------- #
    print("Generating ASCII art...")

    # Convert video to ascii
    convertVideoToAscii(
        filename,
        outFile,
        fps,
        colors,
        frameAutoColor,
        sampleRate,
        startCols,
        endCols,
        scale,
        orderedChars,
        args.invert,
        size=size,
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
