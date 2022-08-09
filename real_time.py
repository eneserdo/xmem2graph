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

from ros_capture import RosRun, shift, getBlended
from grapher import getGraph



def getSingleCenterFromMask(pred, depth):
    """
        If pred has SINGLE segments
    """
    ret, thresh = cv2.threshold(pred*20, 5, 255, 0)
    M = cv2.moments(thresh)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return (cY, cX, depth[cY, cX][0])


def getOrderedCenters(pred, depth, display=False):

    ids = np.unique(pred)[1:]
    centers = {}

    for id in ids:
        centers[id] = getSingleCenterFromMask((pred == id).astype(pred.dtype), depth)
        if display:
            cv2.circle(pred, (centers[id][1], centers[id][0]), 3, 255, -1)
            cv2.putText(pred, f"{centers[id]}", (centers[id][1] - 1, centers[id][0] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)

    return centers


def adapt_mask(mask_name):
    mask_image = np.array(Image.open(mask_name))
    uniques = np.unique(mask_image)
    print(uniques)

    num_obj = len(uniques) - 1

    new_mask = np.zeros_like(mask_image).astype(np.int64)

    for i, m in enumerate(uniques):
        new_mask += (mask_image == m)*i

    mask_image = new_mask.astype(np.uint8).copy()

    print(np.unique(mask_image))
    num_obj = len(np.unique(mask_image)) - 1
    return num_obj, mask_image

def nothing(x):
    pass


def main(argv):
    torch.set_grad_enabled(False)

    # default configuration
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

    device = "cuda"

    print("CUDA version: ",torch.version.cuda)
    network = XMem(config, './XMem/saves/XMem.pth').eval().to(device)

    torch.cuda.empty_cache()

    rosrunner = RosRun("asus")
    shift_x=0
    shift_y=0
    set_mask=False
    record_frames=False
    run_xmem=False
    show_blended=False
    graph_show_labels=True
    i=0
    j=0
    threshold=100
    

    with torch.cuda.amp.autocast(enabled=True):

        for temp_img1, temp_img2 in rosrunner.frameGenerator():
            if j==0:
                start=perf_counter()
            
            if temp_img1 is None:
                break

            print("="*50)
            j=j+1
            
            img1=shift((temp_img1).copy(), shift_x, shift_y)
            img2=shift(temp_img2, -shift_x, -shift_y)


            if set_mask:
                cv2.destroyAllWindows()
                print("Wait for the interactive segmentation screen")
                start_mask=perf_counter() 
                set_mask=False
                cv2.imwrite("./ritm_interactive_segmentation/initial.png", img1)
                subprocess.run("cd ./ritm_interactive_segmentation && ./run.sh", shell=True)
                stop_mask=perf_counter()
                num_objects, mask = adapt_mask('./ritm_interactive_segmentation/mask.png')
                processor = InferenceCore(network, config=config)
                processor.set_all_labels(range(1, num_objects+1)) # consecutive labels
                
                frame_torch, _ = image_to_torch(img1, device=device) # frame:=img1
                mask_torch = index_numpy_to_one_hot_torch(mask, num_objects+1).to(device)
                prediction = processor.step(frame_torch, mask_torch[1:])
                prediction = torch_prob_to_numpy_mask(prediction)
                visualization = overlay_davis(img1, prediction)
                cv2.imshow("vis", visualization)
                run_xmem=True
                cv2.createTrackbar('threshold','vis',0,1000,nothing)

                start=start+(stop_mask-start_mask)
            
            if show_blended:
                blended=getBlended(img1, img2)
                blended = cv2.putText(blended, f"shift x:{2*shift_x}, y: {2*shift_y}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        
            if record_frames:
                raise NotImplementedError
                # i=i+1
                # print("Recording: True")
                # cv2.imwrite(os.path.join(color_dir, f"color_{i:0>6}.png"), img1)
                # cv2.imwrite(os.path.join(depth_dir, f"depth_{i:0>6}.png"), img2)

            
            if run_xmem:
                frame_torch, _ = image_to_torch(img1, device=device)
                prediction = processor.step(frame_torch)
                prediction = torch_prob_to_numpy_mask(prediction)
                centers=getOrderedCenters(prediction, img2, display=True)
                print("Centers:", centers)

                
                threshold = cv2.getTrackbarPos('threshold','vis')
                graphImage=getGraph(centers, prediction, threshold, graph_show_labels)
                    

                
                visualization = overlay_davis(img1, prediction)  # frame:=img1
                visualization = cv2.resize(visualization, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)
                graphImage = cv2.resize(graphImage, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)


                cv2.imshow("graphImage", graphImage)
                cv2.imshow("vis", visualization)
            else:
                cv2.imshow('image2', img2*5)
                cv2.imshow('image1', img1)
            
            time_total=perf_counter()-start
            print("fps:", np.round(j/time_total, 2))

            if show_blended:
                blended = cv2.putText(blended, f"FPS: {np.round(j/time_total, 2)}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                blended = cv2.putText(blended, f"Total: {np.round(time_total/j, 3)} s", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
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
            
            if key==ord('r'):
                record_frames = not record_frames

            if key==ord('m'):
                set_mask = True
            
            if key==ord('b'):
                show_blended = not show_blended

            if key==ord('l'):
                graph_show_labels = not graph_show_labels

                graph_show_labels

            if key==ord('t'):
                shift_y=0
                shift_x=0

        # if os.path.exists("./ritm_interactive_segmentation/initial.png"):
        #     os.remove("./ritm_interactive_segmentation/initial.png")

        # if os.path.exists("./ritm_interactive_segmentation/mask.png"):
        #     os.remove("./ritm_interactive_segmentation/mask.png")


if __name__ == '__main__':
    main(sys.argv)
