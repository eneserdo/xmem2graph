from time import perf_counter
import numpy as np
from math import sqrt

class Timer:
    def __init__(self, name="", queue_length=10):
        self.queue_length=queue_length
        self.queue=[]
        self.start_time=None
        self.end_time=None
        self.name=name
    def start(self):
        self.start_time=perf_counter()
        
    def end_and_print(self, no_print=False):
        elapsed=perf_counter()-self.start_time
        self.queue.append(elapsed)
        val=np.round(1/np.mean(np.array(self.queue)), 2)
        if not no_print:
            print(self.name+"FPS: ", val)
        return val


# ZED K matrix:
# [334.14288330078125, 0.0, 301.9004821777344,
#  0.0, 334.14288330078125, 179.2681884765625,
#   0.0, 0.0, 1.0]

def convert_from_uvd(u, v, d):

    # ZED    
    # cx=301.9004821777344
    # cy=179.2681884765625
    # f=334.14288330078125

    # Kinect
    cx=319.5
    cy=239.5
    f=525.0

    scale=1

    d*=scale

    x_over_z = (cx - u) / f
    y_over_z = (cy - v) / f
    z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    x = x_over_z * z
    y = y_over_z * z
    return x, y, z


class Coord:
    """
    Coordinates to represent a pixel in both real world (x, y, z) and image plane (u, v)
    """
    def __init__(self, u:int, v:int, d:float) -> None:
        self.u=u
        self.v=v
        self.d=d
        self.x, self.y, self.z=convert_from_uvd(u,v,d)
        self.imgCoord=np.array([u, v])
        self.realCoord=np.array([self.x, self.y, self.z])


    def realDistanceTo(self, other) -> float:
        return sqrt(np.sum((self.realCoord - other.realCoord)**2))
    
    # def imgDotProductWith(self, other):
    #     return (self.imgCoord-other.imgCoord)/np.linalg.norm(self.imgCoord-other.imgCoord)
    
    def realUnitVectorTo(self, other) -> np.ndarray:
        return (other.realCoord-self.realCoord)/np.linalg.norm(self.realCoord-other.realCoord)


    def __repr__(self) -> str:
        return f"({round(self.x)}, {round(self.x)}, {round(self.x)})"

    def update_d(self, d) -> None:
        self.d=d
        self.x, self.y, self.z=convert_from_uvd(self.u, self.v, self.d)
        self.realCoord=np.array([self.x, self.y, self.z])



def print_debug(**kwargs):
    print("Error:")
    for key, value in kwargs.items():
        print(f"{key} = {value}")



def arrayAndDepthToCoord(array, depth_map):
    assert array.shape[1]==2

    coord_arr=[]
    
    for a in array:
        coord_arr.append(Coord(a[0], a[1], depth_map[a[0], a[1]][0]))
    
    return coord_arr