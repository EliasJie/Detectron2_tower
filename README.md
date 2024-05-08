

# detectron2_tower
基于detectron2螺母识别

## 1. 建立conda python=3.8
conda create -n detectron2_tower python=3.8

conda activate detectron2_tower

## 2. pip
python3 -m pip install --upgrade pip

pip3 config set global.index-url https://mirror.baidu.com/pypi/simple

## 3.  torch==1.10.0 CUDA 11.1
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

<<<<<<< HEAD
## 4. cd detectorn2目录
python -m pip install -e.

## 5. requirements.txt
=======
## 4. requirements.txt
>>>>>>> f8aa0c34f85e4c09b65e3780bb679ddabb3b651f
pip install -r requirements.txt

requirements.txt 包括：
numpy~=1.24.4
opencv-python~=4.9.0.80
tqdm~=4.66.4
torch~=1.10.0+cu111
pycocotools~=2.0.7
iopath~=0.1.9
pillow~=8.4.0
omegaconf~=2.3.0
PyYAML~=6.0.1
fvcore~=0.1.5.post20221221
tabulate~=0.9.0
torchvision~=0.11.0+cu111
packaging~=24.0
setuptools~=59.5.0
matplotlib~=3.7.5
termcolor~=2.4.0
cloudpickle~=3.0.0
protobuf~=3.19.0
<<<<<<< HEAD
=======

## 5. cd detectorn2目录
python -m pip install -e.


>>>>>>> f8aa0c34f85e4c09b65e3780bb679ddabb3b651f
