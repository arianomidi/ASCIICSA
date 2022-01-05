# ASCIICSA: ACSII Computer Synthesized Art

<!-- ABOUT THE PROJECT -->
## About

There are many image to ASCII converters avaliable but these projects were limited and lacked the customization capabilities needed to create artistics images. 
**ASCIICSA** - ***ACSII Computer Synthesized Art*** - can generate highly customizable ASCII art from an input image or video. Take a look at some of the results and even try it out to create your own ASCII art! 

<!-- [![ASCIICSA][product-screenshot]](https://example.com) -->
![flowers](https://user-images.githubusercontent.com/59782445/148183182-4881b9ec-5d11-4dda-bcff-e37b10c73e04.gif)

### Customization Features:
* Color Selection
  * Select number of shades of Greyscale
  * Select standard 8 or 16 ANSII colors for a retro look 
  * Automatically sample colors from source image
* Text
  * Select the font used by uploading font bitmap
  * Choose characters used 
* Conversion
  * Add filters to get desired output
  * Set sampling methods used
  * Set size and resolution of output image
  * Save raw ANSI text to textfile

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
2. Install project dependencies
   ```sh
   pip install -r requirements.txt
   ```

<!-- USAGE EXAMPLES -->
## Usage

### Convert an image to ASCII image
```sh
  python ascii.py /path/to/image
```
<img src="resources/images/zebra_converted.png"/>
  
### Convert an image to animated ASCII image
```sh
  python ascii_animation.py /path/to/image
```
<img width=500 src="resources/videos/girl.gif"/>

### Convert an image to ASCII video
```sh
  python ascii_video.py /path/to/video
```
![flowers3](https://user-images.githubusercontent.com/59782445/148185944-9aa4ec1a-07e0-435f-b0c8-190f9028f5dc.gif)

### Customization Parameters
```
Optional Arguments:

  -g [GREYSCALESCHEME], --greyscale [GREYSCALESCHEME]
                        Select for greyscale image and pass number of shades used (defaults to true and 8 shades).
  -c [COLORSCHEME], --color [COLORSCHEME]
                        Select for colored image, use with --autoColor for best results (defaults to ANSI16 colors).
  -a, --autoColor       Sample color pallet from most prominent colors in the picture (defalut: 8 samples for grescale and 16 samples for color).
  -n COLS, --cols COLS  The number of characters on the width of the output image (default: 120).
  -l SCALE, --scale SCALE
                        The width-to-height ratio of the pixels sampled for each character (default: 0.6).
  -F FONTPATH, --fontPath FONTPATH
                        The path to the font to be used (default: SFMono-Medium).
  -t CHARS, --chars CHARS
                        The ASCII characters to be used or select from presets: [printable, alphanumeric, alpha, numeric, lower, upper, tech, symbols]
                        (default: printable)
  -i, --invert          Invert the output of the image (default: light characters on black background).
  -r RESOLUTION, --resolution RESOLUTION
                        The resolution of the output image (default: 1920)
  -f CONSTRASTFACTOR, --constrastFactor CONSTRASTFACTOR
                        Contrast factor: <1 less contrast, 1 no change, >1 more contrast (default: 1.3).
  -T {resize,median,mean}, --sampling {resize,median,mean}
                        The sampling method used: [resize, median, mean] (default: resize).
  -C {nearest,fixed}, --colorSelection {nearest,fixed}
                        The color selection method used: [nearest, fixed] (default: nearest).
  -o OUTFILE, --out OUTFILE
                        Output text location.
  -O IMGOUTFILE, --imgout IMGOUTFILE
                        Output image location.
  -H, --hide            Do not open image after conversion (default: false).
  -s, --save            Save ASCII image (default: false).
  -p, --print           Print ASCII text to output (default: false).
```

## Examples

```sh
python3 ascii.py -n 160 woman.jpg    # more characters
```
<img src="resources/images/sof.png"/>

```sh
python3 ascii.py -ac stary_night.jpg    # auto color
```
<img src="resources/images/stary_night.png"/>

***For more examples see the `./resources/` folder.***

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

Arian Omidi - arian.omidi@icloud.com

***Any ideas or improvements are much appreciated.***
