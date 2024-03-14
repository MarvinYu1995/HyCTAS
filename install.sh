pip install torchvision==0.4.0
git clone https://github.com/NVIDIA/apex
cd apex
git checkout f3a960f
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# pip install torch==1.5.0
# pip install timm==0.2.1
pip install torchvision==0.8.2
pip install numba
pip install pycocotools
pip install yacs
pip install tensorboardX
pip install thop==0.0.31.post2001170342
pip install pandas
# pip install --upgrade git+https://github.com/Lyken17/pytorch-OpCounter.git
pip install --upgrade git+https://github.com/sovrasov/flops-counter.pytorch.git
pip install easydict
pip install numpy==1.16.1
pip install matplotlib==3.0.0
pip install pyyaml
pip install tensorboard --upgrade
pip install tqdm
pip install opencv-python==3.4.4.19
pip install Pillow==6.2.0
pip install scipy==1.1.0
pip install protobuf==3.8.0
pip install timm
pip install git+https://github.com/mcordts/cityscapesScripts.git
pip install git+https://github.com/cocodataset/panopticapi.git
# python -m pip install detectron2==0.3 -f \
#   https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.5/index.html
git clone --branch v0.4 https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd ..
# pip install -U iopath
# pip install git+https://github.com/facebookresearch/fvcore.git
pip install einops
# python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.6/index.html
