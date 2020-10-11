# Spacechase-D4PG
HS Bremen Spacechase D4PG Agent implementation

## Dependencies

Python 3.7.7
- tensorflow 2.1.0
- OpenAI Gym 0.17.1 (gym)
- opencv-python 4.2.0.32
- imageio 2.9.0
- dm-reverb 0.1.0 (only linux)
- jupyterlab 2.2.8

## Installation

### Install CUDA on Ubuntu 20.04

Use NVIDIA driver metapackage from nvidia-driver-450 (proprietary)

    sudo apt install nvidia-driver-450
    sudo apt update

Installing CUDA 10.1

    sudo apt install nvidia-cuda-toolkit
    sudo apt update

Installing cuDNN
    
    wget https://developer.download.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.1_20191031/cudnn-10.1-linux-x64-v7.6.5.32.tgz?3T82yUCFTvWSXFaatQPHzGfpshCyQR1jGL9DXqloSYbHvVBxxTzVA96LSpN_IhQJIH7_vtH5K1T9MipbXDxoCitaxWntZBTowa8BpgcodeF6G5gq1E92PkBn6Q-eaElIeVTCu0F74bSTfz0OMiWganSbZLsM-Gz9JpZ2imHvY-eo-y2WQXeMh8Pe7CLSL0d26m65eFNQ_rF7nbDBi7eeICb5JXRg5NUM-Q -O cudnn-10.1-linux-x64-v7.6.5.32.tgz 
    tar -xvzf cudnn-10.1-linux-x64-v7.6.5.32.tgz
    sudo cp cuda/include/cudnn.h /usr/lib/cuda/include/
    sudo cp cuda/lib64/libcudnn* /usr/lib/cuda/lib64/
    sudo chmod a+r /usr/lib/cuda/include/cudnn.h /usr/lib/cuda/lib64/libcudnn*
    sudo rm -r cuda
    sudo rm cudnn-10.1-linux-x64-v7.6.5.32.tgz 

Export CUDA environment variables

    echo 'export LD_LIBRARY_PATH=/usr/lib/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/lib/cuda/include:$LD_LIBRARY_PATH' >> ~/.bashrc

Verification and restart

    nvidia-smi # verify nvidia drivers installed
    nvcc -V # verify CUDA 10.1 installed
    # do restart 

### Create virtualenv
    sudo apt-get install python3-venv
    cd Spacechase-D4PG
    python3 -m venv venv

### Install dependencies
    source venv/bin/activate
    pip install wheel
    pip install tensorflow gym opencv-python imageio dm-reverb jupyterlab

