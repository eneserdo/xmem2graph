from time import perf_counter
import numpy as np
from math import sqrt
import yaml
from easydict import EasyDict


def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return EasyDict(cfg)


# Global variables
cfg=load_config("./config.yml")

if cfg.CAMERA_NAME=="ZED":
    cx=cfg.ZED.cx
    cy=cfg.ZED.cy
    f=cfg.ZED.f
    scale = cfg.ZED.SCALE

elif cfg.CAMERA_NAME=="KINECT":
    cx=cfg.KINECT.cx
    cy=cfg.KINECT.cy
    f=cfg.KINECT.f
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
        elapsed = perf_counter()-self.start_time
        self.queue.append(elapsed)

        if len(self.queue)>self.queue_length:
            self.queue.pop(0)

        val = np.round(1/np.mean(np.array(self.queue)), 2)
        if not no_print:
            print(self.name+"FPS: ", val)
        return val



def convert_from_uvd(u, v, d):
    """
    u, v, d -> x, y, z
    
    +———— x(u)
    |
    |
    y(v)  
    """
    d *= scale

    x_over_z = (u - cx) / f
    y_over_z = (v - cy) / f
    z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    x = x_over_z * z
    y = y_over_z * z
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
        return sqrt(np.sum((self.realCoord - other.realCoord)**2))

    # def imgDotProductWith(self, other):
    #     return (self.imgCoord-other.imgCoord)/np.linalg.norm(self.imgCoord-other.imgCoord)

    def realUnitVectorTo(self, other) -> np.ndarray:
        return (other.realCoord-self.realCoord)/np.linalg.norm(self.realCoord-other.realCoord)

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
    if not ((isinstance(a, float) or isinstance(a, int))):
        print_debug(a=a)
        
    if not ((isinstance(b, float) or isinstance(b, int))):
        print_debug(b=b)

    if abs(a-b)<eps:
        return True
    else:
        False