## Version 2

import os
from os import path
from argparse import ArgumentParser
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset
from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from inference.inference_core import InferenceCore

from progressbar import progressbar

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

device="cuda"

network = XMem(config, './saves/XMem.pth').eval().to(device)

video_name = 'a_output.mp4'
# mask_name = 'a_mask.png'

def getSingleCenterFromMask(pred):
    """
        If pred has SINGLE segments
    """
    ret,thresh = cv2.threshold(pred*20,5,255,0)
    M = cv2.moments(thresh)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return (cX, cY)  

def getOrderedCenters(pred, display=False):

    ids=np.unique(pred)[1:]
    centers={}

    for id in ids:
        centers[id]=getSingleCenterFromMask((pred==id).astype(pred.dtype))
        if display:
            cv2.circle(pred, (centers[id][0], centers[id][1]), 3, 255, -1)
            cv2.putText(pred, f"{centers[id]}", (centers[id][0] - 1, centers[id][1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)

    return centers
        


def adapt_mask(mask_name):
    mask_image = np.array(Image.open(mask_name))
    uniques=np.unique(mask_image)
    print(uniques)

    num_obj = len(uniques) - 1

    new_mask=np.zeros_like(mask_image).astype(np.int64)

    for i, m in enumerate(uniques):
        new_mask+=(mask_image==m)*i

    mask_image=new_mask.astype(np.uint8).copy()

    print(np.unique(mask_image))
    num_obj = len(np.unique(mask_image)) - 1
    return num_obj, mask_image


num_objects, mask= adapt_mask('a_mask.png')
# import os
# os.mkdir("results")


import cv2
from inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis
from time import perf_counter

torch.cuda.empty_cache()

processor = InferenceCore(network, config=config)
processor.set_all_labels(range(1, num_objects+1)) # consecutive labels
cap = cv2.VideoCapture(video_name)

# You can change these two numbers
# frames_to_propagate = 200
visualize_every = 50
avr_time=0
current_frame_index = 0


with torch.cuda.amp.autocast(enabled=True):
    while (cap.isOpened()):
            
            start=perf_counter()
            
            # load frame-by-frame
            _, frame = cap.read()
            if frame is None:
                break

            # convert numpy array to pytorch tensor format
            frame_torch, _ = image_to_torch(frame, device=device)
            if current_frame_index == 0:
                # initialize with the mask
                mask_torch = index_numpy_to_one_hot_torch(mask, num_objects+1).to(device)

                # the background mask is not fed into the model
                prediction = processor.step(frame_torch, mask_torch[1:])
            else:
                # propagate only
                prediction = processor.step(frame_torch)

            # argmax, convert to numpy
            prediction = torch_prob_to_numpy_mask(prediction)
            visualization = overlay_davis(frame, prediction)

            # if current_frame_index>visualize_every:
            #     break

            stop=perf_counter()

            avr_time=avr_time+stop-start

            # cv2.imwrite(f"results/res_{current_frame_index:0>6}.png",visualization)

            cv2.imshow("vis", visualization)
            cv2.waitKey(1)

            # if current_frame_index % visualize_every == 0:
            #   print(f"Frame:{current_frame_index} Total:{np.round(stop-start, 3)}")
            #   visualization = overlay_davis(frame, prediction)
            # display(Image.fromarray(visualization))


            current_frame_index += 1

