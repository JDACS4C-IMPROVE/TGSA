# TGSA
TGSA: Protein-Protein Association-Based Twin Graph Neural Networks for Drug Response Prediction with Similarity Augmentation

# Overview
Here we provide an implementation of Twin Graph neural networks with Similarity Augmentation (TGSA) in Pytorch and PyTorch Geometric. The repository is organised as follows:
Cancel changes
- `data/` contains the necessary dataset files;
- `models/` contains the implementation of TGDRP and SA;
- `TGDRP_weights` contains the trained weights of TGDRP;
- `utils/` contains the necessary processing subroutines;
- `preprocess_gene.py` preprocessing for genetic profiles;
- `smiles2graph.py` construct molecular graphs based on SMILES;
- `main.py main` function for TGDRP (train or test);

## Requirements
- Please install the environment using anaconda3;  
  conda create -n TGSA python=3.7
- Install the necessary packages.
- pip install git+https://github.com/ECP-CANDLE/candle_lib@develop   
  wget https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl   
  wget https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_scatter-2.0.9-cp37-cp37m-linux_x86_64.whl   
  wget https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_sparse-0.6.12-cp37-cp37m-linux_x86_64.whl         
  wget https://data.pyg.org/whl/torch-1.9.0%2Bcu102/torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl   
  pip install *.whl

  pip install rdkit   
  pip install fitlog   

  pip install dgllife==0.3.2   
  pip install dgl==0.9.0   
  pip install numpy==1.21.5   
  pip install pandas==1.3.5   
  pip install scikit-learn==1.0.2   
  pip install scikit-image==0.16.2   
  pip install networkx==2.6.3   
  pip install h5py==3.8.0   

# Implementation
## Step1: Data Preprocessing
  python pilot_preprocessing.py
## Step2: Model Training
  python [candle_train.py](candle_train.py)
## Step3: Model Testing
  python test.py


