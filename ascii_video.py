import cv2
import time
import os, sys, subprocess
from pathlib import Path
from shutil import rmtree
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import argparse

from ansi import *
from ascii import (
    convertImageToAscii,
    autoColor,
    autoSize,
    defaultPalatte,
)
from char_density import *


def convertOpenCV2PIL(img_cv):
    img_pil = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_pil)


def convertPIL2OpenCV(img_pil):
    img_arr = np.array(img_pil)
    return cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)


def extractFrame(filename, out=None, dest=0.5):
    """Extracts frame from video.
    dest=0 -> start
    dest=1 -> end"""

    cam = cv2.VideoCapture(filename)

    ret, frame = cam.read()
    while cam.get(cv2.CAP_PROP_POS_AVI_RATIO) <= dest:
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


def boomerang(filename, out=None, fps=24, progress=True):
    file = Path(str(filename))
    # Check if file given is valid
    if not file.exists():
        print("ERROR: Video does not exist.")
        exit(0)

    frames_path = file.parent / "frames"
    frames_path.mkdir(parents=True, exist_ok=True)

    # create the frames
    vid_to_img(file, frames_path, progress_bar=progress)

    # get frames
    frames = sorted(frames_path.glob("*"))
    # create boomerang effect by looping the frames
    frames.extend(reversed(frames))

    # save frames to video
    out = out or Path("{}/{}_boomerang.mp4".format(file.parent, file.stem))
    img_to_vid(out, fps, frames=frames, progress_bar=progress)

    # delete frame dir
    try:
        rmtree(frames_path)
    except OSError as e:
        print("Error: %s : %s" % (frames_path, e.strerror))


def img_to_vid(out, fps, in_path=None, frames=None, progress_bar=False):
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
    # init video capture
    cam = cv2.VideoCapture(str(filename))
    total_frames = round(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    # init output video
    # TODO: remove for getDefaultSize()
    if not size:
        success, frame = cam.read()
        frame_pil = convertOpenCV2PIL(frame).convert("RGB")  # TODO: change if too slow
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
            frame = convertOpenCV2PIL(frame).convert("RGB")  # TODO: change if too slow

            # auto color every n frames
            if frameAutoColor and frame_num % colorSampleRate == 0:
                colors = frameAutoColor(frame)

            # get number of cols
            video_pos = frame_num / total_frames
            cols = round((endCols - startCols) * video_pos + startCols)

            # convert image to ascii
            ascii_img = convertImageToAscii(
                frame, colors, cols, scale, chars, invert, filter=False, size=size
            )

            # convert ascii frame from PIL to OpenCV and add to the video
            ascii_img_cv = convertPIL2OpenCV(ascii_img)
            video.write(ascii_img_cv)

            frame_num += 1

    # Release all space and windows once done
    cam.release()
    video.release()
    cv2.destroyAllWindows()


def movingAsciiImage(
    image,
    out,
    fps,
    total_frames,
    colors,
    cols,
    scale,
    chars,
    invert,
    size=None,
):

    # init output video
    video = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    image = image.resize(size)

    # set increase
    inc = 2
    image = image.convert("L")
    img_arr = np.array(image).astype(np.int32)
    frame_arr = np.array(image).astype(np.int32)

    max_val = 255
    min_val = 0
    mask = (min_val <= img_arr) & (img_arr <= max_val)
    inc_mask = img_arr <= max_val

    # bg_hue = 0 if invert else 255
    # bg_inc = inc if invert else -inc

    # init progress bar
    frame_num = 0
    with tqdm(
        desc="Creating animated ASCII Image", total=total_frames, ncols=75, unit="f"
    ) as pbar:
        while frame_num < total_frames:
            # update the progress bar
            pbar.update(1)

            # get next frame if available
            np.putmask(frame_arr, mask & inc_mask, frame_arr + inc)
            np.putmask(frame_arr, mask & ~inc_mask, frame_arr - inc)
            # frame_arr = np.where(mask & inc_mask, frame_arr + inc, frame_arr - inc)
            # frame_arr = np.where(mask & dec_mask, frame_arr - inc, frame_arr)
            inc_mask = np.logical_or(frame_arr <= min_val, inc_mask)
            inc_mask = np.logical_and(frame_arr < max_val, inc_mask)
            # dec_mask = np.logical_or(frame_arr >= max_val, dec_mask)
            # dec_mask = np.logical_and(frame_arr > min_val, dec_mask)

            frame_arr = np.clip(frame_arr, 0, 255)

            # convert OpenCV image to PIL and to greyscale
            frame = Image.fromarray(frame_arr).convert(
                "RGB"
            )  # TODO: change if too slow

            # # background color
            # bg_hue += bg_inc
            # if bg_hue <= 0:
            #     bg_inc = -bg_inc
            #     bg_hue = 0
            # elif bg_hue >= 255:
            #     bg_inc = -bg_inc
            #     bg_hue = 255

            # bg_color = (bg_hue, bg_hue, bg_hue)

            # convert image to ascii
            ascii_img = convertImageToAscii(
                frame,
                colors,
                cols,
                scale,
                chars,
                invert,
                filter=True,
                size=size,
            )

            # convert ascii frame from PIL to OpenCV and add to the video
            ascii_img_cv = convertPIL2OpenCV(ascii_img)
            video.write(ascii_img_cv)

            frame_num += 1

    # Release all space and windows once done
    video.release()
    cv2.destroyAllWindows()


def test():
    filename = Path("data/billevans.jpg")
    out = Path("out/moving_image/bill_test_ascii_nearest.mp4")
    image = Image.open(str(filename)).convert("RGB")
    fps = 16

    movingAsciiImage(
        image,
        out,
        fps,
        256,
        defaultPalatte(),
        120,
        0.6,
        True,
        True,
        size=autoSize(image),
    )


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
    parser.add_argument("-a", "--autoColor", dest="autoColor", action="store_true")
    parser.add_argument(
        "-R", "--colorSampleRate", dest="colorSampleRate", type=int, nargs="?", const=-1
    )
    parser.add_argument(
        "-n", "--cols", dest="cols", type=int, nargs="+", required=False
    )
    parser.add_argument("-l", "--scale", dest="scale", type=float, default=0.6)
    parser.add_argument("-m", "--morelevels", dest="moreLevels", action="store_true")
    parser.add_argument("-i", "--invert", dest="invert", action="store_false")
    parser.add_argument("-f", "--fps", dest="fps", type=int, required=False)
    parser.add_argument("-t", "--chars", dest="chars", default=ascii_chars)
    parser.add_argument(
        "-r", "--resolution", dest="resolution", type=int, default=1920, required=False
    )

    parser.add_argument("-o", "--out", dest="outFile", required=False)
    parser.add_argument("-H", "--hide", dest="hide", action="store_true")
    parser.add_argument("-T", "--test", dest="test", action="store_true")

    # parse args
    args = parser.parse_args()

    # open image and convert to grayscale
    filename = Path(args.filename)
    # Check if file given is valid
    if not filename.exists():
        print("ERROR: Video does not exist.")
        exit(0)

    # set text output file
    outFile = Path("out/{}_ascii.mp4".format(filename.stem))
    if args.outFile:
        outFile = Path(args.outFile)

    # set scale default as 0.6 which suits
    # a Courier font
    scale = args.scale

    # set cols
    startCols = 80
    endCols = 80
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

    # test frame
    if args.test:
        # get frames
        start_frame = convertOpenCV2PIL(extractFrame(str(filename), dest=0)).convert(
            "RGB"
        )
        mid_frame = convertOpenCV2PIL(extractFrame(str(filename), dest=0.5)).convert(
            "RGB"
        )
        end_frame = convertOpenCV2PIL(extractFrame(str(filename), dest=1)).convert(
            "RGB"
        )

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

        # plt.figure(figsize=(6, 10))
        # plt.subplot(3, 1, 1)
        # plt.imshow(start_ascii)
        # plt.subplot(3, 1, 2)
        # plt.imshow(start_ascii)
        # plt.subplot(3, 1, 3)
        # plt.imshow(end_ascii)
        # plt.show()

        response = input("Continue? (y/n) ".format(outFile))
        if response.upper() != "Y":
            exit(0)

    # -------------------------------------- #
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

    print("generating ASCII art...")

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
