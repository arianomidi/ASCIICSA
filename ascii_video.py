import cv2
import time
import os, sys, subprocess
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm, trange
import argparse

from ascii import convertImageToAscii, color_schemes, autoColor


def convertOpenCV2PIL(img_cv):
    img_pil = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_pil)


def convertPIL2OpenCV(img_pil):
    img_arr = np.array(img_pil)
    return cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)


def extractFrame(filename, out=None):
    cam = cv2.VideoCapture(filename)
    total_frames = cam.get(7)
    cam.set(1, total_frames // 2)

    ret, frame = cam.read()

    if out:
        cv2.imwrite(out, frame)

    cam.release()
    cv2.destroyAllWindows()

    return frame


def img_to_vid(in_path, out, fps):
    # Get all the frame paths
    frames = sorted(in_path.glob("*"))

    # Get the dimensions of the video
    frame = cv2.imread(str(frames[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    for frame in frames:
        video.write(cv2.imread(str(frame)))

    cv2.destroyAllWindows()
    video.release()


def vid_to_img(filename, out):
    # init video capture and frame number
    cam = cv2.VideoCapture(str(filename))
    currentframe = 0

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
    cols,
    scale,
    moreLevels,
    invert,
):
    # init video capture
    cam = cv2.VideoCapture(str(filename))
    total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    # init output video
    success, frame = cam.read()
    frame_pil = convertOpenCV2PIL(frame).convert("RGB")  # TODO: change if too slow
    ascii_img = convertImageToAscii(frame_pil, colors, cols, scale, moreLevels, invert)
    width, height = ascii_img.size[0], ascii_img.size[1]
    video = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    # init progress bar
    frame_num = 0
    with tqdm(total=total_frames, ncols=75, unit="f") as pbar:
        while success:
            # update the progress bar
            pbar.update(1)

            # convert OpenCV image to PIL and to greyscale
            frame = convertOpenCV2PIL(frame).convert("RGB")  # TODO: change if too slow

            # auto color every n frames
            if frameAutoColor and frame_num % colorSampleRate == 0:
                colors = frameAutoColor(frame)

            # convert image to ascii
            ascii_img = convertImageToAscii(
                frame, colors, cols, scale, moreLevels, invert, filter=False
            )

            # convert ascii frame from PIL to OpenCV and add to the video
            ascii_img_cv = convertPIL2OpenCV(ascii_img)
            video.write(ascii_img_cv)

            # get next frame if available
            success, frame = cam.read()
            if not success:
                break
            frame_num += 1

            # if frame_num <= 3:
            #     ascii_img.show()
            #     frame.show()
            # print(frame_num)
            # frame_num += 1

    # Release all space and windows once done
    cam.release()
    video.release()
    cv2.destroyAllWindows()


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
    parser.add_argument("-n", "--cols", dest="cols", type=int, required=False)
    parser.add_argument("-l", "--scale", dest="scale", type=float, required=False)
    parser.add_argument("-m", "--morelevels", dest="moreLevels", action="store_true")
    parser.add_argument("-i", "--invert", dest="invert", action="store_false")
    parser.add_argument("-f", "--fps", dest="fps", type=int, required=False)

    parser.add_argument("-o", "--out", dest="outFile", required=False)
    parser.add_argument("-H", "--hide", dest="hide", action="store_true")
    parser.add_argument("-t", "--test", dest="test", action="store_true")

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

    # set scale default as 0.5 which suits
    # a Courier font
    scale = 0.5
    if args.scale:
        scale = float(args.scale)

    # set cols
    cols = 80
    if args.cols:
        cols = int(args.cols)

    # set fps
    fps = args.fps
    if not args.fps:
        cam = cv2.VideoCapture(str(filename))
        fps = int(cam.get(cv2.CAP_PROP_FPS))
        cam.release()
        cv2.destroyAllWindows()

    # set sample rate
    sampleRate = fps
    if args.colorSampleRate:
        sampleRate = args.colorSampleRate

    # set color scheme
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
        sample_frame = convertOpenCV2PIL(extractFrame(str(filename))).convert("RGB")
        if frameAutoColor:
            colors = frameAutoColor(sample_frame)
        ascii_img = convertImageToAscii(
            sample_frame, colors, cols, scale, args.moreLevels, args.invert
        )
        ascii_img.show()

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
    start = time.perf_counter()

    # Convert video to ascii
    convertVideoToAscii(
        filename,
        outFile,
        fps,
        colors,
        frameAutoColor,
        sampleRate,
        cols,
        scale,
        args.moreLevels,
        args.invert,
    )

    # Show video
    if not args.hide:
        if sys.platform == "win32":
            os.startfile(outFile)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, outFile])

    end = time.perf_counter()
    print(f"Completed {end - start:0.4f} seconds")
    # # ------------------------------------------------------------------#

    # print("Converting video to frames...")
    # start = time.perf_counter()

    # # Check if file given is valid
    # if not filename.exists():
    #     print("ERROR : file does not exist.")
    #     exit(0)

    # # Create the frame dir if does not exist
    # frames_out.mkdir(parents=True, exist_ok=True)

    # # Convert video to frames
    # vid_to_img(filename, frames_out)
    # print(" - Frames saved at '{}'".format(frames_out))

    # end = time.perf_counter()
    # print(f"Completed {end - start:0.4f} seconds")

    # # ------------------------------------------------------------------#

    # print("Converting frames to ASCII...")
    # start = time.perf_counter()

    # # Create the frame dir if does not exist
    # ascii_out.mkdir(parents=True, exist_ok=True)

    # # Get all the frame paths
    # frames = sorted(frames_out.glob("*"))

    # for frame in frames:
    #     # convert image to ascii txt
    #     aimg, cimg = covertImageToAscii(
    #         str(frame), color_schemes["grayscale_5"], 120, 0.5, True, True
    #     )

    #     # make image from text
    #     image = text_image(aimg, cimg, True)

    #     # save image
    #     img_name = ascii_out / "{}_ascii.png".format(frame.stem)
    #     image.save(img_name)

    # end = time.perf_counter()
    # print(f"Completed {end - start:0.4f} seconds")

    # # ------------------------------------------------------------------#

    # print("Converting frames to video...")
    # start = time.perf_counter()

    # # Convert frames to video
    # img_to_vid(ascii_out, vid_out, fps)

    # end = time.perf_counter()
    # print(f"Completed {end - start:0.4f} seconds")

    # # ------------------------------------------------------------------#


# call main
if __name__ == "__main__":
    main()
