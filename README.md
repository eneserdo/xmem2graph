# xmem2graph

[RITM](https://github.com/saic-vul/ritm_interactive_segmentation) -> gets the initial mask

[XMem](https://github.com/hkchengrex/XMem/) -> tracks the selected objects

Then the code creates a relation graph


## Setting up the environment

Do not forget to install ROS.

```
# Create conda env
conda create -n x2g python=3.8
conda activate x2g

# Install CUDA and cuDNN, and ADD TO THE PATH!
conda install -c conda-forge cudatoolkit=11.3 cudnn=8.4.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# Install requirements
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt

# (Optional) ROS related packages
conda install -c conda-forge ros-rospy rospkg

# To get pre-trained models
./download.sh  
```

OR use yml file:

```
conda env create -f environment.yml
```

# To run 

```
python real_time.py
```
