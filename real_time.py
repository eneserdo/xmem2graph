import os
import sys
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from time import perf_counter
import subprocess

sys.path.append("./XMem")
from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset
from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from inference.inference_core import InferenceCore
from inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis


# My libs
from ros_capture import RosRun, shift, getBlended
from grapher import GraphDrawer, TableRelation
from utils import Timer, convert_from_uvd, Coord, load_config

# Global variables
label_map={1: "hand",
           2: "pan",
           3: "cup",
           4: "spoon",
           5: "foam_balls",
           6: "hand"}

ZOOM_COEF=1.5

cfg=load_config("./config.yml")


def putNames(graphImg, cntrs):
    for id, center in cntrs.items():
        try:
            cv2.putText(graphImg, cfg.LABEL_MAP[str(id)], (center.u - 10, center.v - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,200), 1)    
        except:
            cv2.putText(graphImg, "?", (center.u - 10, center.v - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,200), 1)    


prev_coord=None
def print_coord(event,x,y, flags, param):
    global prev_coord
    if event == cv2.EVENT_LBUTTONDOWN:
        x=x/ZOOM_COEF
        y=y/ZOOM_COEF
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


def _getSingleCenterFromMask(pred, depth):
    """
        If pred has SINGLE segments
    """
    _, thresh = cv2.threshold(pred*20, 5, 255, 0)
    M = cv2.moments(thresh)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])


    # return np.array([cY, cX, depth[cY, cX][0]])
    # return np.array([cY, cX])
    return Coord(cX, cY, depth.item((cY, cX)))
     

def getOrderedCenters(pred, depth):

    ids = np.unique(pred)[1:]
    centers = {}

    for id in ids:
        # centers[id] = getSingleCenterFromMask((pred == id).astype(pred.dtype), depth)
        centers[id] = _getSingleCenterFromMask((pred == id).astype(pred.dtype), depth)
        # if display:
        #     cv2.circle(pred, (centers[id][1], centers[id][0]), 3, 255, -1)
        #     cv2.putText(pred, f"{centers[id]}", (centers[id][1] - 1, centers[id][0] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)

    return centers


def adapt_mask(mask_name):
    """
    XMem requires mask image to have unique values of [0, 1, 2 ... num_obj]
    """
    mask_image = np.array(Image.open(mask_name))
    uniques = np.unique(mask_image)
    # print(uniques)

    num_obj = len(uniques) - 1

    new_mask = np.zeros_like(mask_image).astype(np.int64)

    for i, m in enumerate(uniques):
        new_mask += (mask_image == m)*i

    mask_image = new_mask.astype(np.uint8).copy()

    # print(np.unique(mask_image))
    num_obj = len(np.unique(mask_image)) - 1
    print("Mask is ready")
    return num_obj, mask_image

def nothing(x):
    pass


def main(argv):
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


    device="cuda"

    print("CUDA version: ",torch.version.cuda)
    network = XMem(config, './XMem/saves/XMem.pth').eval().to(device)

    torch.cuda.empty_cache()

    set_mask=False
    mask_ready=False
    try:
        if argv[1]=="mask_ready":
            set_mask=True
            mask_ready=True 
    except IndexError:
        pass

    try:
        shift_x=int(argv[2]) 
        shift_y=int(argv[3])
    except IndexError:
        shift_x=0
        shift_y=0
        pass
    
    rosrunner = RosRun()
    # shift_x=0
    # shift_y=0
    record_frames=False
    run_xmem=False
    show_blended=False
    graph_show_labels=True
    fps=0
    timer=Timer()

    

    with torch.cuda.amp.autocast(enabled=True):

        for temp_rgb, temp_depth in rosrunner.frameGenerator():
            if cfg.CAMERA_NAME=="ZED":
                temp_depth=np.where(np.isnan(temp_depth), 0, temp_depth)
                temp_depth=np.where(temp_depth==np.inf, -1, temp_depth)
                temp_depth=np.where(temp_depth==-1, np.max(temp_depth), temp_depth)
                temp_rgb=temp_rgb[:,:,:-1] # To exclude alpha channel

            timer.start()

            if temp_rgb is None:
                break

            # print("="*50)
            rgb_img=shift((temp_rgb).copy(), shift_x, shift_y)
            depth_img=shift(temp_depth, -shift_x, -shift_y)

            if set_mask:
                cv2.destroyAllWindows()
                set_mask=False

                if not mask_ready:
                    print("Wait for the interactive segmentation screen")
                    cv2.imwrite("./ritm_interactive_segmentation/initial.png", rgb_img)
                    subprocess.run("cd ./ritm_interactive_segmentation && ./run.sh", shell=True)

                num_objects, mask = adapt_mask('./ritm_interactive_segmentation/mask.png')
                processor = InferenceCore(network, config=config)
                processor.set_all_labels(range(1, num_objects+1)) # consecutive labels
                
                frame_torch, _ = image_to_torch(rgb_img, device=device) # frame:=img1
                mask_torch = index_numpy_to_one_hot_torch(mask, num_objects+1).to(device)
                prediction = processor.step(frame_torch, mask_torch[1:])
                prediction = torch_prob_to_numpy_mask(prediction)
                visualization = overlay_davis(rgb_img, prediction)

                cv2.namedWindow("vis")
                cv2.imshow("vis", visualization)

                cv2.namedWindow("graphImage")
                cv2.setMouseCallback('graphImage', print_coord)
                cv2.createTrackbar('check_threshold','graphImage',300,750,nothing)
                cv2.createTrackbar('touch_threshold','graphImage',30,100,nothing)
                cv2.createTrackbar('table_threshold','graphImage',15,100,nothing)
                run_xmem=True

                graphDrawer=GraphDrawer(num_obj=num_objects, beta=0.95)
                #tableRelation=TableRelation(img1, img2, num_objects)
            
            if show_blended:
                blended=getBlended(rgb_img, depth_img)
                blended = cv2.putText(blended, f"shift x:{2*shift_x}, y: {2*shift_y}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        
            if record_frames:

                # np.save("/home/enerd/catkin_ws/rgb_frame_kinect.npy", img1)
                # cv2.imwrite("/home/enerd/catkin_ws/rgb_frame_kinect.png", img1)
                # np.save("/home/enerd/catkin_ws/depth_frame_kinect.npy", img2)
                # cv2.imwrite("/home/enerd/catkin_ws/depth_frame_kinect.png", img2)
                # np.save("/home/enerd/catkin_ws/prediction_frame_kinect.npy", prediction)
                # cv2.imwrite("/home/enerd/catkin_ws/prediction_frame_kinect.png", prediction)
                # print("saved")
                # record_frames=False
                raise NotImplementedError
                # i=i+1
                # print("Recording: True")
                # cv2.imwrite(os.path.join(color_dir, f"color_{i:0>6}.png"), img1)
                # cv2.imwrite(os.path.join(depth_dir, f"depth_{i:0>6}.png"), img2)

            
            if run_xmem:
                frame_torch, _ = image_to_torch(rgb_img, device=device)
                prediction = processor.step(frame_torch)
                prediction = torch_prob_to_numpy_mask(prediction)
                centers=getOrderedCenters(prediction, depth_img)
                # print("Centers:", centers)

                check_threshold = cv2.getTrackbarPos('check_threshold','graphImage')
                touch_threshold = cv2.getTrackbarPos('touch_threshold','graphImage')
                table_threshold = cv2.getTrackbarPos('table_threshold','graphImage')
                
                start_graph=perf_counter()
                graphImage=graphDrawer.getGraph(centers, prediction.copy(), check_threshold, touch_threshold, table_threshold, depth=depth_img, show_labels=graph_show_labels)
                putNames(graphImage, centers)
                print("graph time:", round(perf_counter()-start_graph, 3))
                # graphDrawer.getTableRelations()
                
                visualization = overlay_davis(rgb_img, prediction)  # frame:=img1
                visualization = cv2.resize(visualization, None, fx=ZOOM_COEF, fy=ZOOM_COEF, interpolation=cv2.INTER_CUBIC)
                graphImage = cv2.resize(graphImage, None, fx=ZOOM_COEF, fy=ZOOM_COEF, interpolation=cv2.INTER_CUBIC)

                visualization = cv2.putText(visualization, f"FPS: {fps}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                
                #tableRelation._test(table_threshold, img1, img2)
                
                cv2.imshow("graphImage", graphImage)
                cv2.imshow("vis", visualization)
            else:
                if cfg.CAMERA_NAME=="ZED":
                    depth_img_=((2**16-1)*depth_img/(np.max(depth_img)-np.min(depth_img))).astype(np.uint16)
                elif cfg.CAMERA_NAME=="KINECT":
                    depth_img_=depth_img*20

                cv2.imshow('image2', depth_img_)
                cv2.imshow('image1', rgb_img)
            
            fps=timer.end_and_print(no_print=True)

            if show_blended:
                blended = cv2.putText(blended, f"FPS: {fps}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow('blended', blended)
            
            
            
            key = cv2.waitKey(1) & 0xFF
            
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

            elif key==ord('m'):
                set_mask = True
            
            elif key==ord('u'):
                cfg=load_config("./config.yml")
            
            elif key==ord('b'):
                show_blended = not show_blended
                if not show_blended:
                    cv2.destroyWindow("blended")

            elif key==ord('l'):
                graph_show_labels = not graph_show_labels

            elif key==ord('t'):
                shift_y=0
                shift_x=0

        # if os.path.exists("./ritm_interactive_segmentation/initial.png"):
        #     os.remove("./ritm_interactive_segmentation/initial.png")

        # if os.path.exists("./ritm_interactive_segmentation/mask.png"):
        #     os.remove("./ritm_interactive_segmentation/mask.png")


if __name__ == '__main__':
    depth_img=None
    tableRelation=None
    main(sys.argv)
