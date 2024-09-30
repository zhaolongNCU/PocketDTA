# PocketDTA
Motivation: Accurately predicting the drug-target binding affinity (DTA) is crucial to drug discovery and repurposing. Although deep learning has been widely used in this field, it still faces challenges with insufficient generalization performance, inadequate use of three-dimensional (3D) information and poor interpretability. 
Results: To alleviate these problems, we developed the PocketDTA model. This model enhances the generalization performance by pre-trained models ESM-2 and GraphMVP. It ingeniously handles the first three (top-3) target binding pockets and drug 3D information through customized GVP-GNN Layers and GraphMVP-Decoder. Additionally, it employs a bilinear attention network to enhance interpretability. Com-parative analysis with state-of-the-art (SOTA) methods on the optimized Davis and KI-BA datasets reveals that the PocketDTA model exhibits significant performance ad-vantages. Further, ablation studies confirm the effectiveness of the model components, whereas cold-start experiments illustrate its robust generalization capabilities. In particu-lar, the PocketDTA model has shown significant advantages in identifying key drug functional groups and amino acid residues via molecular docking and literature valida-tion, highlighting its strong potential for interpretability. 

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
Furthermore, our code is based on python 3.7 and CUDA 11.7 and is used on a linux system. Note that while our code does not require a large amount of running memory, it is best to use more than 24G of RAM if you need to run it.
## Dataset
The two benchmark datasets Davis and KIBA have been placed in the dataset folder `process.csv`, and each folder also includes the `GraphMVP.pth` pre-training model parameter file and the top1-top3 target-binding pockets of 3-dimensional information`.pickle` file.The target 3D structure `.pdb files` for the two benchmark datasets, along with the first three pocket `.pdb files`, are available for download on [Google Cloud Drive](https://drive.google.com/drive/folders/1qJXsxkTSgwPSTpu-XmIUh2rD2jJ1KuGQ).
## Pre-trained model
Since the parameter files for the other pre-trained models are rather large, we will not give them here, you can download them according to the link below and save them to the appropriate location in the PocketDTA folder. [ESM-2](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt),[ProtBert](https://zenodo.org/records/4633691),[ProtT5](https://zenodo.org/records/4644188),[GraphMVP](https://github.com/chao1224/GraphMVP),[3Dinfomax](https://github.com/HannesStark/3DInfomax).

## Training
Once you have configured the base environment and dataset as well as some pre-trained models, you are ready to train the models.

```
python --task Davis --r 0
```
In the meantime you can run the .sh file on a Linux system and train different seeds.

```
./training.sh
```
## Ablation study
**Representation ablation study**

You can run the .sh file on a Linux system.
```
./Ablation.sh
```
**Module ablation study**

```
./Ablation_module.sh
```
## Cold study
```
./Cold.sh
```
## Interpretability analysis
Firstly you need to change the data to the sample you want to test and then run the code below to get the drug weights atomic counterpart and amino acid residue counterpart weights.

```
python interaction_weight.py --task Davis --model DTAPredictor_test --r 2 --use-test True
```
