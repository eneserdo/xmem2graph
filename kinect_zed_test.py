import rospy
import numpy as np
from sensor_msgs.msg import Image as Img_Msg
from ros_capture import shift
import cv2
import sys

class RosRun:
    def __init__(self, camera="all"):
        """
        Example Usage:
        
        """
        self.camera=camera
        if camera=="all":
            self.img_topic1 = "/camera/rgb/image_rect_color"
            self.img_topic2 = "/camera/depth_registered/sw_registered/image_rect_raw"
            self.img_topic3 = "/zed/zed_node/rgb/image_rect_color"
            self.img_topic4 = "/zed/zed_node/depth/depth_registered"
            
        else:
            raise ValueError("Unknown camera!")
        
        self.temp_img1 = None
        self.temp_img2 = None
        self.temp_img3 = None
        self.temp_img4 = None
        self.img_ready1 = False
        self.img_ready2 = False
        self.img_ready3 = False
        self.img_ready4 = False

    
    def imgCallback1(self, data):
        self.temp_img1 = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        self.img_ready1 = True

    def imgCallback2(self, data):
        self.temp_img2 = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height, data.width)    
        self.img_ready2 = True

    def imgCallback3(self, data):
        self.temp_img3 = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        self.img_ready3 = True

    def imgCallback4(self, data):
        self.temp_img4 = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width)    
        self.img_ready4 = True


    def frameGenerator(self):
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber(self.img_topic1, Img_Msg, self.imgCallback1)
        rospy.Subscriber(self.img_topic2, Img_Msg, self.imgCallback2)
        rospy.Subscriber(self.img_topic3, Img_Msg, self.imgCallback3)
        rospy.Subscriber(self.img_topic4, Img_Msg, self.imgCallback4)
        print("Waiting...")
        

        while not rospy.is_shutdown():       
            if self.img_ready1 and self.img_ready2 and self.img_ready3 and self.img_ready4:
                self.img_ready1=False
                self.img_ready2=False
                self.img_ready3=False
                self.img_ready4=False
                yield self.temp_img1, self.temp_img2, self.temp_img3, self.temp_img4
            
        
        yield None, None, None, None


def convert_from_uvd_z(u, v, d):
    # ZED
    cx=301.9004821777344
    cy=179.2681884765625
    f=334.14288330078125

    scale = 1000
    d *= scale

    x_over_z = (u - cx) / f
    y_over_z = (v - cy) / f
    z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    x = x_over_z * z
    y = y_over_z * z
    return x, y, z

def convert_from_uvd_k(u, v, d):
    # Kinect
    cx = 319.5
    cy = 239.5
    f = 525.0

    scale = 1
    d *= scale

    x_over_z = (u - cx) / f
    y_over_z = (v - cy) / f
    z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    x = x_over_z * z
    y = y_over_z * z
    return x, y, z



prev_coord_k=None
def print_coord_k(event,x,y, flags, param):
    global prev_coord_k
    if event == cv2.EVENT_LBUTTONDOWN:
        print("x, y, d: ",round(x),round(y), img2.item((int(y),int(x))))
        x1,y1,z1=convert_from_uvd_k(x, y, img2.item((int(y),int(x))))
        
        print(f"({round(x1, 2)}, {round(y1, 2)}, {round(z1, 2)})")	
        
        if prev_coord_k is not None:
            print("dist: ", np.round(np.sqrt(np.sum((prev_coord_k-np.array([x1, y1, z1]))**2))))
        prev_coord_k=np.array([x1, y1, z1])


prev_coord_z=None
def print_coord_z(event,x,y, flags, param):
    global prev_coord_z
    if event == cv2.EVENT_LBUTTONDOWN:
        print("x, y, d: ",round(x), round(y), temp_img4.item((int(y),int(x))))
        x1,y1,z1=convert_from_uvd_z(x, y, temp_img4.item((int(y),int(x))))
        print(f"({round(x1, 5)}, {round(y1, 5)}, {round(z1, 5)})")	
        if prev_coord_z is not None:
            print("dist: ", np.round(np.sqrt(np.sum((prev_coord_z-np.array([x1, y1, z1]))**2)), 5))
        prev_coord_z=np.array([x1, y1, z1])

def main(argv):
    global img2, temp_img4

    rosrunner = RosRun("all")
    shift_x=0
    shift_y=0
    fps=0

    cv2.namedWindow("rgb_k")
    cv2.setMouseCallback('rgb_k', print_coord_k)

    cv2.namedWindow("rgb_z")
    cv2.setMouseCallback('rgb_z', print_coord_z)


    for temp_img1, temp_img2, temp_img3, temp_img4 in rosrunner.frameGenerator():
        if temp_img1 is None:
            break

        temp_img4=np.where(np.isnan(temp_img4), 0, temp_img4)
        temp_img4=np.where(temp_img4==np.inf, -1, temp_img4)
        temp_img4=np.where(temp_img4==-1, np.max(temp_img4), temp_img4)

        temp_img3=temp_img3[:,:, :-1]
        

        img1=shift((temp_img1).copy(), shift_x, shift_y)
        img2=shift(temp_img2, -shift_x, -shift_y)
        cv2.imshow('rgb_k', img1)

        #img2=((2**16-1)*img2/(np.max(img2)-np.min(img2))).astype(np.uint16)
        cv2.imshow('depth_k', img2*20)



        
        cv2.imshow('rgb_z', temp_img3)

        cv2.imshow('depth_z', ((2**16-1)*temp_img4/(np.max(temp_img4)-np.min(temp_img4))).astype(np.uint16))
        
        
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

        if key==ord('t'):
            shift_y=0
            shift_x=0

if __name__ == '__main__':
    img2=None
    temp_img4=None
    main(sys.argv)
