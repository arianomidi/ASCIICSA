import cv2
import time
import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm, trange

from ascii import convertImageToAscii, text_image, color_schemes


def convertOpenCV2PIL(img_cv):
    img_pil = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_pil)


def convertPIL2OpenCV(img_pil):
    img_arr = np.array(img_pil)
    return cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)


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


def convertVideoToAscii(filename, out):
    fps = 18
    colors = color_schemes["grayscale_5"]
    cols = 80
    scale = 0.5
    moreLevels = True
    invert = True

    # init video capture
    cam = cv2.VideoCapture(str(filename))
    total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cam.get(cv2.CAP_PROP_FPS))

    # init output video
    success, frame = cam.read()
    frame_pil = convertOpenCV2PIL(frame).convert("L")
    ascii_img = convertImageToAscii(frame_pil, colors, cols, scale, moreLevels, invert)
    width, height = ascii_img.size[0], ascii_img.size[1]
    video = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    # init progress bar
    with tqdm(total=total_frames, ncols=75, unit="f") as pbar:
        while success:
            # update the progress bar
            pbar.update(1)

            # convert OpenCV image to PIL and to greyscale
            frame = convertOpenCV2PIL(frame).convert("L")

            # convert image to ascii
            ascii_img = convertImageToAscii(
                frame, colors, cols, scale, moreLevels, invert
            )

            # convert ascii frame from PIL to OpenCV and add to the video
            ascii_img_cv = convertPIL2OpenCV(ascii_img)
            video.write(ascii_img_cv)

            # get next frame if available
            success, frame = cam.read()
            if not success:
                break

            # if frame_num <= 3:
            #     ascii_img.show()
            #     frame.show()
            # print(frame_num)
            # frame_num += 1

    # Release all space and windows once done
    cam.release()
    video.release()
    cv2.destroyAllWindows()


def test():
    filename = Path("data/test_video.mp4")
    out = Path("out/video/test1.mp4")

    frames_out = Path("out/video/frames/")
    ascii_out = Path("out/video/ascii/")
    vid_out = Path("out/video/eyes_ascii.mp4")
    fps = 18

    print("Converting video to ascii...")
    start = time.perf_counter()

    # Check if file given is valid
    if not filename.exists():
        print("ERROR: Video does not exist.")
        exit(0)

    # Convert video to ascii
    convertVideoToAscii(filename, out)

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
    test()
