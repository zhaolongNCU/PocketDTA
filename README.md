# PocketDTA
PocketDTA: an advanced multimodal architecture for enhanced prediction of drug-target affinity from 3D structural data of target binding pockets

## Architecture
![PocketDTA](https://github.com/zhaolongNCU/PocketDTA/blob/main/PocketDTA.jpg)

## Installation
First, you need to clone our code to your operating system.

```
git clone https://github.com/zhaolongNCU/PocketDTA.git
cd PocketDTA
```


## The environment of PocketDTA
Before running the code, you need to configure the environment, which mainly consists of the commonly used torch==1.13.0+cu117, rdkit==2023.3.2, torch-geometric==2.3.1 and other basic packages.
```
python==3.7.16
torch==1.13.0+cu117
torch-geometric==2.3.1
scipy==1.7.3
rdkit==2023.3.2
pandas==1.3.5
ogb==1.3.5
networkx==2.6.3
mol2vec==0.2.2
fair-esm==2.0.0
h5py==3.8.0
dgl==1.1.3
```
Of course you can also directly use the following to create a new environment:
```
conda create -n PocketDTA python==3.7
conda activate PocketDTA
pip install requirements.txt
```
where the requirements.txt file is already given in the code.
Furthermore, our code is based on python 3.7 and CUDA 11.7 and is used on a linux system configured with an NVIDIA A800-80G PCIE GPU and an Intel_5318Y 2.1GHz CPU. Note that while our code does not require a large amount of running memory, it is best to use more than 24G of RAM if you need to run it.
## Dataset
The two benchmark datasets Davis and KIBA have been placed in the dataset folder `process.csv`, and each folder also includes the `GraphMVP.pth` pre-training model parameter file and the top1-top3 target-binding pockets of 3-dimensional information`.pickle` file.The target 3D structure `.pdb files` for the two benchmark datasets, along with the first three pocket `.pdb files`, are available for download on [Google Cloud Drive](https://drive.google.com/drive/folders/1qJXsxkTSgwPSTpu-XmIUh2rD2jJ1KuGQ).
## pre-trained model
Since the parameter files for the other pre-trained models are rather large, we will not give them here, you can download them according to the link below and save them to the appropriate location in the PocketDTA folder.

## Training
Once you have configured the base environment and dataset as well as some pre-trained models, you are ready to train the models.

```
python --task Davis --r 0
```
In the meantime you can run the .sh file on a Linux system and train different seeds.

```
./training.sh
```
