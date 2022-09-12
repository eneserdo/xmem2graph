from typing import Optional, Tuple
from time import perf_counter
from easydict import EasyDict
from math import sqrt
import numpy as np
import argparse
import yaml
import cv2


def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return EasyDict(cfg)


# Global variables
cfg = load_config("./config.yml")

if cfg.CAMERA_NAME == "ZED":
    cx = cfg.ZED.cx
    cy = cfg.ZED.cy
    f = cfg.ZED.f
    scale = cfg.ZED.SCALE

elif cfg.CAMERA_NAME == "KINECT":
    cx = cfg.KINECT.cx
    cy = cfg.KINECT.cy
    f = cfg.KINECT.f
    scale = cfg.KINECT.SCALE


class Timer:
    def __init__(self, name="", queue_length=10):
        self.queue_length = queue_length
        self.queue = []
        self.start_time = None
        self.end_time = None
        self.name = name

    def start(self):
        self.start_time = perf_counter()

    def end_and_print(self, no_print=False):
        elapsed = perf_counter() - self.start_time
        self.queue.append(elapsed)

        if len(self.queue) > self.queue_length:
            self.queue.pop(0)

        val = np.round(1 / np.mean(np.array(self.queue)), 2)
        if not no_print:
            print(self.name + "FPS: ", val)
        return val


def convert_from_uvd(u, v, d):
    """
    u, v, d -> x, y, z

    +———— x(u)
    |
    |
    y(v)
    """
    # FALSE!!
    # d *= scale
    # x_over_z = (u - cx) / f
    # y_over_z = (v - cy) / f
    # z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    # x = x_over_z * z
    # y = y_over_z * z

    z = d * scale
    x = (u - cx) * z / f
    y = (v - cy) * z / f

    return x, y, z


class Coord:
    """
    Coordinates to represent a pixel in both real world (x, y, z) and image plane (u, v).
    Notice that u//x and v//y
    """

    def __init__(self, u: int, v: int, d: float) -> None:
        self.u = u
        self.v = v
        self.d = d

        if not (isinstance(u, float) or isinstance(u, int)):
            print("Erorr")
            print(f"{u=}, {type(u)=}")
            raise ValueError

        if not (isinstance(v, float) or isinstance(v, int)):
            print("Erorr")
            print(f"{v=}, {type(v)=}")
            raise ValueError

        if not (isinstance(d, float) or isinstance(d, int)):
            print("Erorr")
            print(f"{d=}, {type(d)=}")
            raise ValueError

        self.x, self.y, self.z = convert_from_uvd(u, v, d)
        self.imgCoord = np.array([u, v])
        self.realCoord = np.array([self.x, self.y, self.z])

    def realDistanceTo(self, other) -> float:
        return sqrt(np.sum((self.realCoord - other.realCoord) ** 2))

    # def imgDotProductWith(self, other):
    #     return (self.imgCoord-other.imgCoord)/np.linalg.norm(self.imgCoord-other.imgCoord)

    def realUnitVectorTo(self, other) -> np.ndarray:
        return (other.realCoord - self.realCoord) / np.linalg.norm(
            self.realCoord - other.realCoord
        )

    def __repr__(self) -> str:
        return f"({round(self.x)}, {round(self.y)}, {round(self.z)})"

    def update_d(self, d) -> None:
        self.d = d
        self.x, self.y, self.z = convert_from_uvd(self.u, self.v, self.d)
        self.realCoord = np.array([self.x, self.y, self.z])


def print_debug(**kwargs):
    print("Error:")
    for key, value in kwargs.items():
        print(f"{key} = {value}, type: {type(value)}")


def arrayAndDepthToCoord(array, depth_map):
    assert array.shape[1] == 2
    assert len(depth_map.shape) == 2
    # array consists of (v,u) pairs.
    coord_arr = []

    for a in array:
        # a[0]=v, a[1]=u
        coord_arr.append(Coord(a.item(1), a.item(0), depth_map.item((a[0], a[1]))))

    return coord_arr


def almost_equal(a, b, eps=1e-5):
    # assert (isinstance(a, float) or isinstance(a, int))
    # assert (isinstance(b, float) or isinstance(b, int))
    # if not ((isinstance(a, float) or isinstance(a, int))):
    #     print_debug(a=a)

    # if not ((isinstance(b, float) or isinstance(b, int))):
    #     print_debug(b=b)

    if abs(a - b) < eps:
        return True
    else:
        False


def argparser():
    # camera name
    # help="These are dynamic parameters, for others, check 'config.yml' "
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-mr", "--mask_ready", action="store_true", help="to load already existing mask"
    )
    ap.add_argument(
        "-sm", "--set_mask", action="store_true", help="to initialize RITM directly"
    )

    ap.add_argument(
        "-tpr",
        "--table_points_ready",
        action="store_true",
        help="to load already existing points",
    )

    ap.add_argument("-p", "--profile", action="store_true", help="run with cProfile")

    ap.add_argument("-r", "--record_frames", action="store_true", help="record")

    ap.add_argument(
        "-mn", "--mask_name", default="mask.png", help="to load already existing mask"
    )
    ap.add_argument(
        "-sx",
        "--shift_x",
        type=int,
        default=0,
        help="shift in x axis to align depth with rgb",
    )
    ap.add_argument(
        "-sy",
        "--shift_y",
        type=int,
        default=0,
        help="shift in y axis to align depth with rgb",
    )

    return ap.parse_args()


def dictToText(d):
    text = ""
    for key, value in d.items():
        text = text + str(key) + ": " + str(round(value)) + "\n"
    return text


def getMinMax(depth):
    h, w = depth.shape
    x, y, z = convert_from_uvd(0, 0, 4)

    max_x, max_y, max_z = -float("infinity"), -float("infinity"), -float("infinity")
    min_x, min_y, min_z = float("infinity"), float("infinity"), float("infinity")

    for v in range(h):
        for u in range(w):
            d = depth.item((v, u))

            if almost_equal(d, 0):
                continue
            x, y, z = convert_from_uvd(u, v, d)

            min_x = x if min_x > x else min_x
            min_y = y if min_y > y else min_y
            min_z = z if min_z > z else min_z

            max_x = x if max_x < x else max_x
            max_y = y if max_y < y else max_y
            max_z = z if max_z < z else max_z

    return np.array([min_x, min_y, min_z]), np.array([max_x, max_y, max_z])


def writeCenters(i, centers_dict, max_, min_):
    for id_, center in centers_dict.items():
        name = cfg.LABEL_MAP[id_]

        normalized = (center.realCoord() - min_) / (max_ - min_)

        with open(f"center_{name}.txt", "a") as f:
            f.writelines(
                f"{i}: {normalized.item(0), normalized.item(1), normalized.item(2)}"
                + "\n"
            )


# Borrowed from https://stackoverflow.com/a/73471951/10006394
def add_text_to_image(
    image_rgb: np.ndarray,
    label: str,
    top_left_xy: Tuple = (0, 0),
    font_scale: float = 1,
    font_thickness: float = 1,
    font_face=cv2.FONT_HERSHEY_SIMPLEX,
    font_color_rgb: Tuple = (0, 0, 255),
    bg_color_rgb: Optional[Tuple] = None,
    outline_color_rgb: Optional[Tuple] = None,
    line_spacing: float = 1,
):
    """
    Adds text (including multi line text) to images.
    You can also control background color, outline color, and line spacing.

    outline color and line spacing adopted from: https://gist.github.com/EricCousineau-TRI/596f04c83da9b82d0389d3ea1d782592
    """
    OUTLINE_FONT_THICKNESS = 3 * font_thickness

    im_h, im_w = image_rgb.shape[:2]

    for line in label.splitlines():
        x, y = top_left_xy

        # ====== get text size
        if outline_color_rgb is None:
            get_text_size_font_thickness = font_thickness
        else:
            get_text_size_font_thickness = OUTLINE_FONT_THICKNESS

        (line_width, line_height_no_baseline), baseline = cv2.getTextSize(
            line,
            font_face,
            font_scale,
            get_text_size_font_thickness,
        )
        line_height = line_height_no_baseline + baseline

        if bg_color_rgb is not None and line:
            # === get actual mask sizes with regard to image crop
            if im_h - (y + line_height) <= 0:
                sz_h = max(im_h - y, 0)
            else:
                sz_h = line_height

            if im_w - (x + line_width) <= 0:
                sz_w = max(im_w - x, 0)
            else:
                sz_w = line_width

            # ==== add mask to image
            if sz_h > 0 and sz_w > 0:
                bg_mask = np.zeros((sz_h, sz_w, 3), np.uint8)
                bg_mask[:, :] = np.array(bg_color_rgb)
                image_rgb[
                    y : y + sz_h,
                    x : x + sz_w,
                ] = bg_mask

        # === add outline text to image
        if outline_color_rgb is not None:
            image_rgb = cv2.putText(
                image_rgb,
                line,
                (x, y + line_height_no_baseline),  # putText start bottom-left
                font_face,
                font_scale,
                outline_color_rgb,
                OUTLINE_FONT_THICKNESS,
                cv2.LINE_AA,
            )
        # === add text to image
        image_rgb = cv2.putText(
            image_rgb,
            line,
            (x, y + line_height_no_baseline),  # putText start bottom-left
            font_face,
            font_scale,
            font_color_rgb,
            font_thickness,
            cv2.LINE_AA,
        )
        top_left_xy = (x, y + int(line_height * line_spacing))

    return image_rgb
