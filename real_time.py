import os
import sys
import cv2
import rospy
import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
import numpy as np
from time import perf_counter, localtime, time
import subprocess
import glob
from datetime import datetime
import random


sys.path.append("./XMem")
# from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset
# from inference.data.mask_mapper import MaskMapper

from model.network import XMem
from inference.inference_core import InferenceCore
from inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis


# from _Float64 import Float64
# from rospy.numpy_msg import numpy_msg


from std_msgs.msg import Float64MultiArray


# My libs
from ros_capture import RosRun
# from ros_capture import shift, getBlended
from grapher import GraphDrawer, TableRelation
from utils import Timer, convert_from_uvd, Coord, load_config, argparser, add_text_to_image, dictToText, putTwoColumnText, serialize_graph_raw
from Mask import mask2orderedCenters, adaptMask

import cProfile

# ZOOM_COEF=1.2

cfg=load_config("./config.yml")


def putNames(graphImg, cntrs, name_centers_offset={}, construct=[True]):
    if construct[0]:
        for id_, center in cntrs.items():
            r=np.random.randint(20, 90)
            minus_or_plus=random.choice([1, -1])
            theta=np.random.rand()*(np.pi*2/3)+np.pi/6

            theta *= minus_or_plus

            name_centers_offset[id_]=[round(r*np.cos(theta)), round(-r*np.sin(theta))]

        construct[0]=False

    for id_, center in cntrs.items():

        name_center=(center.u+name_centers_offset[id_][0], center.v+ name_centers_offset[id_][1])
        try:
            cv2.line(graphImg, (center.u, center.v), name_center, (255,60,200), 1)
            cv2.putText(graphImg, cfg.LABEL_MAP[str(id_)], name_center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,60,200), 1)    
        except:
            cv2.putText(graphImg, "?", (center.u - 10, center.v - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,60,200), 1)    


prev_coord=None
def print_coord(event,x,y, flags, param):
    global prev_coord
    if event == cv2.EVENT_LBUTTONDOWN:
        # x=x/ZOOM_COEF
        # y=y/ZOOM_COEF
        print(round(x),round(y))
        x1,y1,z1=convert_from_uvd(x, y, depth_img.item((int(y),int(x))))
        print(f"({round(x1, 2)}, {round(y1, 2)}, {round(z1, 2)})")	
        if prev_coord is not None:
            print("dist: ", np.round(np.sqrt(np.sum((prev_coord-np.array([x1, y1, z1]))**2))))
            try:
                print("distToTable: ", tableRelation._distToPoint(x1,y1,z1))
            except:
                pass

        prev_coord=np.array([x1, y1, z1])



def nothing(x):
    pass


def main(args):
    global depth_img, tableRelation, cfg
    torch.set_grad_enabled(False)

    # default configuration for XMem
    config = {
        'top_k': 30,
        'mem_every': 5,
        'deep_update_every': -1,
        'enable_long_term': True,
        'enable_long_term_count_usage': True,
        'num_prototypes': 128,
        'min_mid_term_frames': 5,
        'max_mid_term_frames': 10,
        'max_long_term_elements': 10000,
    }


    assert torch.cuda.is_available()
    device="cuda"


    print("CUDA version: ", torch.version.cuda)
    network = XMem(config, './XMem/saves/XMem.pth').eval().to(device)

    torch.cuda.empty_cache()

    mask_ready = args.mask_ready
    set_mask = args.mask_ready or args.set_mask
    
    shift_x = args.shift_x
    shift_y = args.shift_y


    rosrunner = RosRun()
    record_frames=args.record_frames
    current_time=datetime.now()
    fname=os.path.join("results", args.output_folder+"-"+ current_time.strftime("%m-%d-%Y, %H-%M-%S"))
    
    if record_frames:
        os.makedirs(fname)
        # files = glob.glob('/results/*')
        # if files:
        #     for f in files:
        #         os.remove(f)

    run_xmem=False
    show_blended=False
    fps=0
    timer=Timer()
    i=1

    pub = rospy.Publisher('graph_raw', Float64MultiArray, queue_size=10)
    # pub = rospy.Publisher('graph_raw', numpy_msg(Float64), queue_size=10)


    with torch.cuda.amp.autocast(enabled=True):

        counter=0

        # FOR ROS BAGS
        for temp_rgb, temp_depth in rosrunner.frameGenerator():
            if cfg.CAMERA_NAME=="ZED":

                # remove nans
                temp_depth=np.where(np.isnan(temp_depth), 0, temp_depth)

                # remove posinfs
                max_val=temp_depth[temp_depth!=np.inf].max()
                temp_depth=np.where(temp_depth==np.inf, max_val, temp_depth)

                # remove neginfs
                min_val=temp_depth[temp_depth!=-np.inf].min()
                temp_depth=np.where(temp_depth==-np.inf, min_val, temp_depth)
                
                temp_rgb=temp_rgb[:,:,:-1] # To exclude alpha channel
            
            # np.save("temp_depth.npy", temp_depth)
            # break

        # # # # # # # # # #

        # FOR IMAGE FOLDERS
        # rgb_files=sorted(glob.glob("/home/enerd/xmem2graph/rosdata/aralik_cut/rgb/*.png"))
        # depth_files=sorted(glob.glob("/home/enerd/xmem2graph/rosdata/aralik_cut/depth/*.npy"))
        # # rgb_files=sorted(glob.glob("rosdata/aralik_stir/rgb/*.png"))
        # # depth_files=sorted(glob.glob("rosdata/aralik_stir/depth/*.npy"))

        # for rgb_path, depth_path in zip(rgb_files, depth_files):
        #     temp_rgb=cv2.imread(rgb_path, -1)
        #     temp_rgb=cv2.resize(temp_rgb, (640, 360))
            # temp_depth=cv2.imread(depth_path, -1)
            # counter+=1
            # print(counter, "/", len(rgb_files))

            # temp_rgb=np.load(rgb_path)
            # temp_depth=np.load(depth_path)
        # # # # # # # # # #

            timer.start()

            if temp_rgb is None:
                break
            
            if shift_x!=0 or shift_x!=0:
                # rgb_img=shift(temp_rgb, shift_x, shift_y)
                # depth_img=shift(temp_depth, -shift_x, -shift_y)
                pass
            else:
                rgb_img=temp_rgb
                depth_img=temp_depth
            
            if set_mask:
                cv2.destroyAllWindows()
                set_mask=False

                if not mask_ready:
                    print("Wait for the interactive segmentation screen")
                    cv2.imwrite("./ritm_interactive_segmentation/initial.png", rgb_img)
                    subprocess.run("cd ./ritm_interactive_segmentation && ./run.sh", shell=True)

                num_objects, mask = adaptMask('./ritm_interactive_segmentation/' + args.mask_name)
                processor = InferenceCore(network, config=config)
                processor.set_all_labels(range(1, num_objects+1)) # consecutive labels
                
                frame_torch, _ = image_to_torch(rgb_img, device=device) # frame:=img1
                mask_torch = index_numpy_to_one_hot_torch(mask, num_objects+1).to(device)
                prediction = processor.step(frame_torch, mask_torch[1:])
                prediction = torch_prob_to_numpy_mask(prediction)
                visualization = overlay_davis(rgb_img, prediction)

                # cv2.namedWindow("vis")
                # cv2.imshow("vis", visualization)

                cv2.namedWindow("combined")
                cv2.setMouseCallback('combined', print_coord)
                cv2.createTrackbar('distance_equality_threshold','combined',100,750,nothing)
                cv2.createTrackbar('visible_relation_id','combined',0,14,nothing)
                run_xmem=True

                graphDrawer=GraphDrawer(num_obj=num_objects, fname=fname, beta=0.95)
            
            if show_blended:
                # blended=getBlended(rgb_img, depth_img)
                # blended = cv2.putText(blended, f"shift x:{2*shift_x}, y: {2*shift_y}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                pass    
            
            if run_xmem:
                frame_torch, _ = image_to_torch(rgb_img, device=device)
                prediction = processor.step(frame_torch)
                prediction = torch_prob_to_numpy_mask(prediction)
                centers=mask2orderedCenters(prediction, depth_img)
                # print("Centers:", centers)
                distance_equality_threshold = cv2.getTrackbarPos('distance_equality_threshold','combined')
                visible_relation_id = cv2.getTrackbarPos('visible_relation_id','combined')
 
                start_graph=perf_counter()
                graphImage, distance_dict, stabile_distance_matrix, ssg = graphDrawer.getGraph(rgb_img, centers, prediction.copy(), distance_equality_threshold, visible_relation_id, depth=depth_img)
                
                # graphDrawer._testProximity(rgb_img, depth_img, (prediction==1)*1, visible_relation_id)

                putNames(graphImage, centers)
                print("graph time:", round(perf_counter()-start_graph, 3))
                
                visualization = overlay_davis(rgb_img, prediction)  # frame:=img1
                # visualization = cv2.putText(visualization, f"FPS: {fps}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                

                # tableRelation._test(table_threshold, rgb_img, depth_img)
               
                # visualization = cv2.resize(visualization, None, fx=ZOOM_COEF, fy=ZOOM_COEF, interpolation=cv2.INTER_CUBIC)
                # graphImage = cv2.resize(graphImage, None, fx=ZOOM_COEF, fy=ZOOM_COEF, interpolation=cv2.INTER_CUBIC)
               
                # Combines graph, simple graph and segmented image
                h1,w1,c1=graphImage.shape
                h2, w2, c2=ssg.shape
                combined=np.zeros((2*h1, w1+w2, c1), dtype=np.uint8)
                
                combined[:h1, :w1]=visualization
                combined[h1:, :w1]=graphImage
                # combined[h1-h2//2 : h1-h2//2+h2, w1:]=ssg
                combined[2*h1-h2:, w1:]=ssg
                # putTwoColumnText(combined, distance_dict, w1, w2)


                cv2.imshow("combined", combined)

                if record_frames:
                    cv2.imwrite(os.path.join(fname, f"combined_{i:0>5}.png"), combined)
                    print(i)
                    i+=1

            else:
                if cfg.CAMERA_NAME=="ZED":
                    depth_img_=((2**16-1)*depth_img/(np.max(depth_img)-np.min(depth_img))).astype(np.uint16)
                elif cfg.CAMERA_NAME=="KINECT":
                    depth_img_=depth_img*20

                depth_img_ = cv2.resize(depth_img_, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                rgb_img = cv2.resize(rgb_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                cv2.imshow('depth_img', depth_img_)
                cv2.imshow('rgb_img', rgb_img)
            
            fps=timer.end_and_print(no_print=True)

            if show_blended:
                blended = cv2.putText(blended, f"FPS: {fps}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow('blended', blended)
            
            # print(depth_img.shape)
            # print(getMinMax(depth_img))

            stabile_distance_matrix = np.arange(12).reshape(2, 2, 3)
            ros_data=Float64MultiArray()
            ros_data.data.extend(serialize_graph_raw(np.array([2, 3]), stabile_distance_matrix))
            # stabile_distance_matrix=np.array([0,1,2,3.]).reshape(2, 2, 1)

            pub.publish(ros_data)

            key = cv2.waitKey(30) & 0xFF
            #
            if key  == ord('q'): 
                break 

            elif key==ord('d'):
                if shift_x<160:
                    shift_x=shift_x+1
            
            elif key==ord('a'):
                if shift_x>-160:
                    shift_x=shift_x-1
                    
            elif key==ord('s'):
                if shift_y<30:
                    shift_y=shift_y+1
                
            elif key==ord('w'):
                if shift_y>-30:
                    shift_y=shift_y-1
            
            elif key==ord('r'):
                record_frames = not record_frames

            elif key==ord('p'):
                cv2.imwrite("color_frame.png", graphImage+15)
                depth_img_=((2**16-1)*depth_img/(np.max(depth_img)-np.min(depth_img))).astype(np.uint16)
                cv2.imwrite("depth_frame.png", depth_img_)
                subprocess.run("/home/enerd/Desktop/playground/opencv/cv_test color_frame.png depth_frame.png && pcl_viewer pcl_file.pcd", shell=True)
                
                try:
                    os.remove("./color_frame.png")
                    os.remove("./depth_frame.png")
                    os.remove("./pcl_file.pcd")
                except:
                    pass

            elif key==ord('m'):
                set_mask = True
                mask_ready = False

            elif key==ord('n'):
                putNames(graphImage, centers, construct=[True])
                
            
            elif key==ord('u'):
                cfg=load_config("./config.yml")
                graphDrawer.update_cfg()
            
            elif key==ord('b'):
                show_blended = not show_blended
                if not show_blended:
                    cv2.destroyWindow("blended")

            elif key==ord('t'):
                shift_y=0
                shift_x=0
            
            elif key==32: # space keycode
                key=0
                print("press space to continue")
                while key != 32:
                    key=cv2.waitKey(0) & 0xFF
                


        # if os.path.exists("./ritm_interactive_segmentation/initial.png"):
        #     os.remove("./ritm_interactive_segmentation/initial.png")

        # if os.path.exists("./ritm_interactive_segmentation/mask.png"):
        #     os.remove("./ritm_interactive_segmentation/mask.png")

        if os.path.exists("./simpleStableGraph.png"):
            os.remove("./simpleStableGraph.png")
    
    


if __name__ == '__main__':
    depth_img=None
    tableRelation=None
    args=argparser()

    assert not ((args.set_mask is True) and (args.mask_ready is True))
    
    if args.profile:
        try:
            cProfile.run("main(args)", "output.prof")
        finally:
            cv2.destroyAllWindows()
            subprocess.run("snakeviz output.prof", shell=True)
    else:
        main(args)
