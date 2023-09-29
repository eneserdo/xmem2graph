from typing import List, Optional, Tuple
from time import perf_counter
from easydict import EasyDict
from math import ceil, floor, isinf, sqrt
import numpy as np
import argparse
import yaml
import cv2
import os


def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return EasyDict(cfg)


# Global variables
cfg = load_config("./config.yml")

if cfg.CAMERA_NAME == "ZED":
    CX = cfg.ZED.cx
    CY = cfg.ZED.cy
    F = cfg.ZED.f
    SCALE = cfg.ZED.SCALE

elif cfg.CAMERA_NAME == "KINECT":
    CX = cfg.KINECT.cx
    CY = cfg.KINECT.cy
    F = cfg.KINECT.f
    SCALE = cfg.KINECT.SCALE


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


def convert_from_uvd(u, v, d, is_np=False):
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
    
    # z = d * scale
    # x = (u - cx) * z / f
    # y = (v - cy) * z / f

    d_= d * SCALE
    x = (u - CX) * d_ / F
    y = - (v - CY) * d_ / F
    z = - d_

    if is_np:
        return np.stack((x, y, z), axis=1)
    else:
        return x, y, z



def convert_from_xyz(x, y, z, is_np=False):
    """
    x, y, z -> u, v, d
    """

    if is_np:
        assert np.isinf(x).any() == False and np.isinf(y).any() == False and np.isinf(z).any() == False
        assert np.isnan(x).any() == False and np.isnan(y).any() == False and np.isnan(z).any() == False
    else:
        raise NotImplementedError("what are you doing?")    

    u = -F * x / z + CX
    v = F * y / z + CY


    mask_to_nan = (u < 0) | (u > cfg.IMG.WIDTH) | (v < 0) | (v > cfg.IMG.HEIGHT)     

    if mask_to_nan.any():
        print("nan vals")

    u[mask_to_nan] = np.nan
    v[mask_to_nan] = np.nan

    if is_np:
        return np.stack((u, v), axis=1)
    else:
        return u, v

class Coord:
    """
    Coordinates to represent a pixel in both real world (x, y, z) and image plane (u, v).
    Notice that u//x and v//y
    """

    def __init__(self, u: int, v: int, d: float) -> None:
        self.u = u
        self.v = v
        self.d = d

    #     if not (isinstance(u, float) or isinstance(u, int)):
    #         print("Erorr")
    #         print(f"{u=}, {type(u)=}")
    #         raise ValueError

    #     if not (isinstance(v, float) or isinstance(v, int)):
    #         print("Erorr")
    #         print(f"{v=}, {type(v)=}")
    #         raise ValueError

    #     if not (isinstance(d, float) or isinstance(d, int)):
    #         print("Erorr")
    #         print(f"{d=}, {type(d)=}")
    #         raise ValueError

    #     self.x, self.y, self.z = convert_from_uvd(u, v, d)
    #     self.imgCoord = np.array([u, v])
    #     self.realCoord = np.array([self.x, self.y, self.z])

    # def realDistanceTo(self, other) -> float:
    #     return sqrt(np.sum((self.realCoord - other.realCoord) ** 2))

    # # def imgDotProductWith(self, other):
    # #     return (self.imgCoord-other.imgCoord)/np.linalg.norm(self.imgCoord-other.imgCoord)

    # def realUnitVectorTo(self, other) -> np.ndarray:
    #     return (other.realCoord - self.realCoord) / np.linalg.norm(
    #         self.realCoord - other.realCoord
    #     )

    # def __repr__(self) -> str:
    #     return f"({round(self.x)}, {round(self.y)}, {round(self.z)})"

    # def update_d(self, d) -> None:
    #     self.d = d
    #     self.x, self.y, self.z = convert_from_uvd(self.u, self.v, self.d)
    #     self.realCoord = np.array([self.x, self.y, self.z])


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

def record_hand_centers(all_centers):

    pass
    # left_id=[i for i, v in enumerate(cfg.LABEL_MAP) if v=="left"][0]
    # right_id=[i for i, v in enumerate(cfg.LABEL_MAP) if v=="right"][0]

    # left_center=all_centers[left_id]
    # right_center=all_centers[right_id]
    
    # right_file="right.txt"
    # left_file="left.txt"

    # if not os.path.isfile(right_file):
    #     with open(right_file, "a") as f:
    #         f.write("x y z" + "\n")

    # if not os.path.isfile(left_file):
    #     with open(left_file, "a") as f:
    #         f.write("x y z" + "\n")

    # with open(right_file, "a") as f:
    #     f.write(f"{right_center[0]} {right_center[1]} {right_center[2]}" + "\n")

    # with open(left_file, "a") as f:
    #     f.write(f"{left_center[0]} {left_center[1]} {left_center[2]}" + "\n")



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
        "-nt", "--no_table", action="store_true", help="to disable table"
    )

    ap.add_argument(
        "-tpr",
        "--table_points_ready",
        action="store_true",
        help="to load already existing points",
    )

    ap.add_argument("-p", "--profile", action="store_true", help="run with cProfile")

    ap.add_argument("-r", "--record_frames", action="store_true", help="record")
    ap.add_argument("-o", "--output_folder", default=".",    help="output folder")

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
   
    texts=[]

    for key, value in d.items():
        textline = ""

        if isinf(value):
            val="inf"
        else:
            val=str(round(value))

        textline = textline + str(key) + ": " + val + "\n"
        texts.append(textline)

    return texts


def putTwoColumnText(image, distance_dict, w1, w2):
    texts=dictToText(distance_dict)
    len_=len(texts)
    max_len=11

    if len_>max_len:
        first_column="".join(texts[:max_len])
        second_column="".join(texts[max_len:])
        image=add_text_to_image(image, first_column, (w1+10, 10), 0.6, font_color_rgb=(255,255,255), bg_color_rgb=(0, 0, 0))
        image=add_text_to_image(image, second_column, (w1+10+w2//2, 10), 0.6, font_color_rgb=(255,255,255), bg_color_rgb=(0, 0, 0))
    else:
        first_column="".join(texts)
        image=add_text_to_image(image, first_column, (w1+10, 10), 0.6, font_color_rgb=(255,255,255), bg_color_rgb=(0, 0, 0))




# def getMinMax(depth):
#     h, w = depth.shape
#     x, y, z = convert_from_uvd(0, 0, 4)

#     max_x, max_y, max_z = -float("infinity"), -float("infinity"), -float("infinity")
#     min_x, min_y, min_z = float("infinity"), float("infinity"), float("infinity")

#     for v in range(h):
#         for u in range(w):
#             d = depth.item((v, u))

#             if almost_equal(d, 0):
#                 continue
#             x, y, z = convert_from_uvd(u, v, d)

#             min_x = x if min_x > x else min_x
#             min_y = y if min_y > y else min_y
#             min_z = z if min_z > z else min_z

#             max_x = x if max_x < x else max_x
#             max_y = y if max_y < y else max_y
#             max_z = z if max_z < z else max_z

#     return np.array([min_x, min_y, min_z]), np.array([max_x, max_y, max_z])


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


def getRotationMatrix(w0: np.ndarray, theta: float = None):
    """Calculates rotation matrix using Rodrigue's formula (norm of the w is assigned to theta)"""
    # FIXME modifies the w implicitly

    if theta is None:
        norm = np.linalg.norm(w0)
        w0 = w0 / norm
        theta = norm
    else:
        assert np.allclose(np.linalg.norm(w0), 1.0)

    W_hat = np.array([[0, -w0[2], w0[1]], [w0[2], 0, -w0[0]], [-w0[1], w0[0], 0]])

    # Rodrigue's formula
    R = (np.identity(3) + np.sin(theta) * W_hat + (1 - np.cos(theta)) * np.matmul(W_hat, W_hat))

    # sanity check
    assert np.allclose(np.linalg.det(R), 1.0)

    return R


def homogeneousMultiplication(A, x):
    """A.shape = [D, D], x.shape = [N, D-1];
    Calculates x' = A @ [x.T, 1].T
    then returns x' in inhomogeneous vector form
    """
    assert A.shape[0] == A.shape[1] and x.shape[1] == (A.shape[0] - 1)

    x = np.concatenate((x, np.ones((x.shape[0], 1))), 1)
    x = A @ x.T
    x = x[:-1] / x[-1]

    return x.T


def rotate_points_in_x(points, theta=-20):
    """points.shape = [N, 3]"""
    R = getRotationMatrix(np.array([1, 0, 0]), theta)
    A = np.identity(4)
    A[:3, :3] = R

    return homogeneousMultiplication(A, points)


def get_bboxed_image(image, bbox_list):
    # bbox_list: N x 6 : [x0, x1, y0, y1, z0, z1]
    # bbox_list -> vertices_xyz -> vertices_uv -> edges

    vertex_pairs = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]

    for bbox in bbox_list:
        if np.allclose(bbox, 0):
            continue

        vertices_xyz = np.array([[bbox[0], bbox[2], bbox[4]], 
                                 [bbox[0], bbox[2], bbox[5]], 
                                 [bbox[0], bbox[3], bbox[4]], 
                                 [bbox[0], bbox[3], bbox[5]], 
                                 [bbox[1], bbox[2], bbox[4]], 
                                 [bbox[1], bbox[2], bbox[5]], 
                                 [bbox[1], bbox[3], bbox[4]], 
                                 [bbox[1], bbox[3], bbox[5]]])
                              
        vertices_uv = convert_from_xyz(vertices_xyz[:, 0], vertices_xyz[:, 1], vertices_xyz[:, 2], is_np=True)
        
        # get edges from vertices

        for v1, v2 in vertex_pairs:
            if np.isnan(vertices_uv[v1]).any() or np.isnan(vertices_uv[v2]).any():
                print("Nan in bbox vertices!")
                continue

            # p1=np.flip(vertices_uv[v1].astype(np.int32))
            # p2=np.flip(vertices_uv[v2].astype(np.int32))
            p1=vertices_uv[v1].astype(np.int32)
            p2=vertices_uv[v2].astype(np.int32)
            try:
                # np.save("image_debug.npy", image)
                # np.save("p1_debug.npy", p1)
                # np.save("p2_debug.npy", p2)
                # print("p1:", p1, "p2:", p2)
                image = cv2.line(image, p1, p2, (0, 0, 100), 1)
            except:
                # raise ValueError()
                print("ERR in line")
                print("v1:", v1, "v2:", v2)
                print("p1:", p1, "p2:", p2)
                print("p1_xyz:", vertices_xyz[v1], "p2_xyz:", vertices_xyz[v2])
                raise ValueError            


def serialize_graph_raw(object_list: np.ndarray, graph_tensor: np.ndarray) -> List:
    # TODO: deal with missing objects
    # object_list: not one-hot-encoded

    assert len(graph_tensor.shape) == 3
    
    assert graph_tensor.shape[0] == object_list.shape[0], "object_list is not matching with graph_tensor"
    
    serialized=np.concatenate([graph_tensor.shape[:1], object_list, graph_tensor.reshape(-1)])
    
    return serialized.tolist()
