import rospy
import numpy as np
from sensor_msgs.msg import Image as Img_Msg
import sys

# sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import cv2
# sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')
     

class RosRun:
    def __init__(self, camera="asus"):
        """
        Example Usage:
        
        """

        if camera=="kinect" or camera=="kinect1":
            self.img_topic1 = "/camera/rgb/image_rect_color"
            self.img_topic2 = "/camera/depth_registered/sw_registered/image_rect_raw"
        
        elif camera=="kinect2":
            self.img_topic1 = "/kinect2/qhd/image_color_rect"
            self.img_topic2 = "/kinect2/qhd/image_depth_rect"
        
        # elif camera=="kinect1" or camera=="kinect":
        #     self.img_topic1 = "/camera_front/rgb/image_color"  # /camera_front/depth_registered/sw_registered/image_rect_raw
        #     self.img_topic2 = "/camera_front/depth/image_raw"

        elif camera=="zed":
            self.img_topic1 = "/zed/zed_node/rgb/image_rect_color"
            self.img_topic2 = "/zed/zed_node/depth/depth_registered"
            
        else:
            raise ValueError("Unknown camera!")
        
        self.temp_img1 = None
        self.temp_img2 = None
        self.img_ready1 = False
        self.img_ready2 = False

    
    def imgCallback1(self, data):
        self.temp_img1 = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)    
        self.img_ready1 = True

    def imgCallback2(self, data):
        self.temp_img2 = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height, data.width)    
        self.img_ready2 = True


    def frameGenerator(self):
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber(self.img_topic1, Img_Msg, self.imgCallback1)
        rospy.Subscriber(self.img_topic2, Img_Msg, self.imgCallback2)
        print("Waiting...")
        

        while not rospy.is_shutdown():       
            if self.img_ready1 and self.img_ready2:
                self.img_ready1=False
                self.img_ready2=False
                yield self.temp_img1, self.temp_img2
            
        
        yield None, None

def getBlended(img1, img2):

    im1=(np.sum(img1,2)//3).astype(np.uint8)
    im1=cv2.merge([np.zeros_like(im1),np.zeros_like(im1), im1])
                
    #clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #im2=clahe.apply(img2)

    im2=cv2.normalize(img2*10, None, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
    im2=im2.astype(np.uint8)
    #im2=np.sum(img2,2).astype(np.float32)
    im2=cv2.merge([im2, np.zeros_like(im2),np.zeros_like(im2)])

    return im2+im1

def shift(img, ox, oy):
    non = lambda s: s if s<0 else None
    mom = lambda s: max(0,s)

    shift_img = np.zeros_like(img)
    shift_img[mom(oy):non(oy), mom(ox):non(ox)] = img[mom(-oy):non(-oy), mom(-ox):non(-ox)]
    return shift_img