from time import perf_counter
import numpy as np

class Timer:
    def __init__(self, name="", queue_length=10):
        self.queue_length=queue_length
        self.queue=[]
        self.start_time=None
        self.end_time=None
        self.name=name
    def start(self):
        self.start_time=perf_counter()
        
    def end_and_print(self):
        elapsed=perf_counter()-self.start_time
        self.queue.append(elapsed)
        val=np.round(1/np.mean(np.array(self.queue)), 2)
        print(self.name+"FPS: ", val)
        return val
        

def rgb_to_real(u, v, d):
    f=525
    cx=320
    cy=240


    return x, y, z