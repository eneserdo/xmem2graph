#!/bin/bash

# eval "$(conda shell.bash hook)"

# conda activate ia_ritm
 
python3 demo.py --checkpoint=./weights/coco_lvis_h32_itermask.pth --limit-longest-size=400

# conda deactivate
