import cv2
import rospy
import numpy as np
from utils import load_config
# from sensor_msgs.msg import Image as Img_Msg

# from cv_bridge import CvBridge
 

cfg=load_config("./config.yml")

class RosRun:
    def __init__(self):
        """
        Example Usage:
        
        """
        if cfg.CAMERA_NAME=="KINECT":
            self.rgb_topic = cfg.KINECT.RGB_TOPIC
            self.depth_topic = cfg.KINECT.DEPTH_TOPIC
        
        elif cfg.CAMERA_NAME=="ZED":
            self.rgb_topic = "/zed/zed_node/rgb/image_rect_color"
            self.depth_topic = "/zed/zed_node/depth/depth_registered"
    
        else:
            raise ValueError("Unknown camera!")
        
        # elif camera=="kinect2":
        #     self.img_topic1 = "/kinect2/qhd/image_color_rect"
        #     self.img_topic2 = "/kinect2/qhd/image_depth_rect"
        
        # elif camera=="kinect1" or camera=="kinect":
        #     self.img_topic1 = "/camera_front/rgb/image_color"  # /camera_front/depth_registered/sw_registered/image_rect_raw
        #     self.img_topic2 = "/camera_front/depth/image_raw"

        
        self.temp_rgb = None
        self.temp_depth = None
        self.rgb_ready = False
        self.depth_ready = False
        # self.br=CvBridge()

    
    def imgCallback1(self, data):
        self.temp_rgb = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        # self.temp_img1 = self.br.imgmsg_to_cv2(data, "passthrough")    
        self.rgb_ready = True

    def imgCallback2(self, data):
        if cfg.CAMERA_NAME=="ZED":
            self.temp_depth = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width)    
        elif cfg.CAMERA_NAME=="KINECT":
            self.temp_depth = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height, data.width)    
        # self.temp_img2 = self.br.imgmsg_to_cv2(data, "passthrough")
        self.depth_ready = True


    def frameGenerator(self):
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber(self.rgb_topic, Img_Msg, self.imgCallback1)
        rospy.Subscriber(self.depth_topic, Img_Msg, self.imgCallback2)
        print("Waiting...")
        

        while not rospy.is_shutdown():       
            if self.rgb_ready and self.depth_ready:
                self.rgb_ready=False
                self.depth_ready=False
                yield self.temp_rgb, self.temp_depth
            
        
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