# # Random color scheme
# std_rgb = [
#   '#000000', '#cc0000', '#4e9a06', '#c4a000', '#729fcf', '#75507b', '#06989a', '#d3d7cf',
#   '#555753', '#ef2929', '#8ae234', '#fce94f', '#32afff', '#ad7fa8', '#34e2e2', '#ffffff'
# ]

# Custom
std_rgb = [
    "#2e3436",
    "#cc0000",
    "#4e9a06",
    "#c4a000",
    "#3465a4",
    "#75507b",
    "#06989a",
    "#d3d7cf",
    "#555753",
    "#ef2929",
    "#8ae234",
    "#fce94f",
    "#729fcf",
    "#ad7fa8",
    "#34e2e2",
    "#eeeeec",
    "#000000",
    "#eeeeec",
]

# # Ubuntu default color scheme
# std_rgb = [
#   '#2e3436', '#cc0000', '#4e9a06', '#c4a000', '#3465a4', '#75507b', '#06989a', '#d3d7cf',
#   '#555753', '#ef2929', '#8ae234', '#fce94f', '#729fcf', '#ad7fa8', '#34e2e2', '#eeeeec',
#   '#300a24', "#eeeeec"
# ]

# # VSCode Dark+ color scheme
# std_rgb = [
#   '#000000', '#cd3131', '#0dbc79', '#e5e510', '#2472c8', '#bc3fbc', '#11a8cd', '#e5e5e5',
#   '#666666', '#f14c4c', '#23d18b', '#f5f543', '#3b8eea', '#d670d6', '#29b8db', '#e5e5e5',
#   "#0e0e0e", "#cccccc"
# ]


def ansi8_rgb():
    ansi8_hex = std_rgb[0:8]
    ansi8_rgb = []

    for hex in ansi8_hex:
        hex_code = hex.lstrip("#")
        rgb = tuple(int(hex_code[2 * i : 2 * i + 2], 16) for i in range(3))
        ansi8_rgb.append(rgb)

    return ansi8_rgb


def ansi16_rgb():
    ansi16_hex = std_rgb[0:16]
    ansi16_rgb = []

    for hex in ansi16_hex:
        hex_code = hex.lstrip("#")
        rgb = tuple(int(hex_code[2 * i : 2 * i + 2], 16) for i in range(3))
        ansi16_rgb.append(rgb)

    return ansi16_rgb


def rgbToAnsi256(r, g, b):
    # we use the extended greyscale palette here, with the exception of
    # black and white. normal palette only has 4 greyscale shades.
    if r == g and g == b:
        if r < 8:
            ansi_num = 16
        elif r > 248:
            ansi_num = 231
        else:
            ansi_num = round(((r - 8) / 247) * 24) + 232
    else:
        ansi_num = (
            16
            + (36 * round(r / 255 * 5))
            + (6 * round(g / 255 * 5))
            + round(b / 255 * 5)
        )

    ansi_code = "\033[38;5;{}m".format(ansi_num)

    return ansi_code


def hexToAnsi256(hex):
    r = int(hex[1:3])
    b = int(hex[3:5])
    g = int(hex[5:])

    return rgbToAnsi256(r, g, b)


def ansi_rgb(ansi):
    """
    Takes raw ansi code and hexcolor code.
    """
    ansi = parse_ansi(ansi)

    if ansi < 0 or ansi > 255:
        return "#000000"
    if ansi < 16:
        return std_rgb[ansi]

    if ansi > 231:
        s = (ansi - 232) * 10 + 8
        rgb = "#%02x%02x%02x" % (s, s, s)
        return rgb

    index_R = (ansi - 16) // 36
    r = 55 + index_R * 40 if index_R > 0 else 0
    index_G = ((ansi - 16) % 36) // 6
    g = 55 + index_G * 40 if index_G > 0 else 0
    index_B = (ansi - 16) % 6
    b = 55 + index_B * 40 if index_B > 0 else 0

    rgb = "#%02x%02x%02x" % (r, g, b)
    return rgb


def parse_ansi(ansi):
    """
    Takes raw ansi code and returns 8-bit color code.
    """
    _, raw_ansi = ansi.split("[")
    codes = raw_ansi.split(";")

    if codes[0] == "38" and codes[1] == "5":  # if 8-bit ansi color code return color
        return int(codes[-1][:-1])
    else:
        end_code = int(codes[-1][:-1])
        if (
            30 <= end_code and end_code <= 37
        ):  # if 3-bit standard ansi color code return 8-bit color code
            return end_code - 30
        elif (
            90 <= end_code and end_code <= 97
        ):  # if 3-bit bright ansi color code return 8-bit color code
            return end_code - 82
        else:
            return -1


def background_color(inverted):
    if inverted:
        return std_rgb[16]
    else:
        return std_rgb[17]
