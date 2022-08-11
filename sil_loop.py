import os
import sys
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# My libs
from ros_capture import RosRun, shift, getBlended
from grapher import GraphDrawer
from utils import Timer

def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y, img2[int(y),int(x)])

        A=np.array([1/525.0, 0, -64/105.0,0, 1/525.0, -16/35,0,0,1]).reshape(3,3)

        print(img2[int(y),int(x)]/((A@np.array([[x], [y], [1]]))[2]))	
        

rosrunner = RosRun("asus")
shift_x=0
shift_y=0
record_frames=False
run_xmem=False
show_blended=False


cv2.namedWindow('image1')
cv2.setMouseCallback('image1',draw_circle)


for temp_img1, temp_img2 in rosrunner.frameGenerator():
    if temp_img1 is None:
        break
    
    img1=shift((temp_img1).copy(), shift_x, shift_y)
    img2=shift(temp_img2, -shift_x, -shift_y)
    
    if show_blended:
        blended=getBlended(img1, img2)
        blended = cv2.putText(blended, f"shift x:{2*shift_x}, y: {2*shift_y}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('image2', img2*5)
    cv2.imshow('image1', img1)
    
    if show_blended:
        blended = cv2.putText(blended, f"FPS: {fps}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('blended', blended)
    

    
    key = cv2.waitKey(1) & 0xFF
    
    if key  == ord('q'): 
        break 

    if key==ord('d'):
        if shift_x<160:
            shift_x=shift_x+1
    
    if key==ord('a'):
        if shift_x>-160:
            shift_x=shift_x-1
            
    if key==ord('s'):
        if shift_y<30:
            shift_y=shift_y+1
        
    if key==ord('w'):
        if shift_y>-30:
            shift_y=shift_y-1
    
    
    if key==ord('b'):
        show_blended = not show_blended

    if key==ord('t'):
        shift_y=0
        shift_x=0
