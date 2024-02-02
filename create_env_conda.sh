#!/usr/bin/env bash

env=$1  # name of the virtual env.

echo "Deleting existing env $env"
conda remove --name $env --all
rm -rf ~/anaconda3/envs/$env

echo "Create  env $env"
conda create -n $env python=3.10

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate $env

python --version

cdir=$(pwd)


echo "Installing..."

pip install -r requirements.txt

pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

pip install pretrainedmodels zipp timm efficientnet-pytorch  kornia

pip install tensorflow
pip install git+https://github.com/rcmalli/keras-vggface.git
pip install keras_applications

pip install facetorch
pip install deepface
pip install seaborn

pip install gdown
pip install pykeops==2.1.2

cd $cdir

if [ ! -d "SynergyNet" ]; then
  git clone https://github.com/choyingw/SynergyNet
  cd SynergyNet
  # pull my pr: https://github.com/choyingw/SynergyNet/pull/27
  git pull origin pull/27/head
  rm -fr .git
  # downloads
  # 1. aflw2000_data.zip [not necessary]

  # gdown 1YVBRcXmCeO1t5Bepv67KVr_QKcOur3Yy
  # unzip -o aflw2000_data.zip
  # rm aflw2000_data.zip

  # 2. 3dmm_data.zip
  gdown 1SQsMhvAmpD1O8Hm0yEGom0C0rXtA0qs8
  unzip -o 3dmm_data.zip
  rm 3dmm_data.zip

  # pretrained/best.pth.tar
  cd pretrained
  gdown 1BVHbiLTfX6iTeJcNbh-jgHjWDoemfrzG
  cd $cdir
fi

# root
cd $cdir

# action units repo: Facial Action Unit Detection With Transformers.
if [ ! -d "FAU_CVPR2021" ]; then
  git clone https://github.com/rakutentech/FAU_CVPR2021.git
  cd FAU_CVPR2021
  git checkout 0bfb778526908f36b6136e836d8b382877bacfa4
  rm -fr .git
  # fold 1: Transformer_FAU_fold0.h5
  gdown 1Wk9e78TVXMF0aQ4b4c0SovIJoJOMgHL5
  # fold 2: Transformer_FAU_fold1.h5
  gdown 1-uwkTp0WS-C-yRFBrm9sPl9WROAWaGUv
  # fold 3: Transformer_FAU_fold2.h5
  gdown 1p7hlZ3sxhbKoZjcACPVTeago1CESLZQ_
  # fe1.png
  gdown 1ZAHzRAnUB8JU0j6kVhgUZ5qA5x53XLo8
  # some test data: BP4D_fold0_10031.npz
  gdown 1AmTfWS0SnkPhQT9DxAfCCIZ53z_PbGk7


  # dependencies: only on gsys: conda create -n FAU_CVPR2021  python=3.7
  # run it separately.
  # Never run this.
  if [ 0 -gt 1 ]; then
    pip install tensorflow-gpu==1.15.0 imutils opencv-python h5py==2.10.0 \
    Keras==2.2.4 keras-pos-embd keras-multi-head keras-layer-normalization \
    keras-position-wise-feed-forward keras-embed-sim pandas scikit-learn \
    matplotlib
    pip install protobuf==3.20.0
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
    pip install pynvml pretrainedmodels efficientnet_pytorch timm
    cd $cdir/dlib/crf/crfwrapper/bilateralfilter
    swig -python -c++ bilateralfilter.i
    python setup.py install
  fi
  # cudnn: required: v7.6.0
  # https://github.com/tensorflow/tensorflow/issues/35376#issuecomment-571737108

fi

cd $cdir

cd $cdir/SynergyNet
# root.
cd Sim3DR
./build_sim3dr.sh
cd ../FaceBoxes
./build_cpu_nms.sh
cd ..
# root.
# install
pip install -e .

cd $cdir

cd $cdir/dlib/crf/crfwrapper/bilateralfilter
swig -python -c++ bilateralfilter.i
python setup.py install

conda deactivate

echo "Done creating and installing virt.env: $env."