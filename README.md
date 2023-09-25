# TGSA
TGSA: Protein-Protein Association-Based Twin Graph Neural Networks for Drug Response Prediction with Similarity Augmentation

Here we provide an implementation of Twin Graph neural networks with Similarity Augmentation (TGSA) in Pytorch and PyTorch Geometric. 
To search for the optimal hyperparameters of the model, we made following changes based on the original repository: 
1. Rewrite the data preprocessing steps to introduce extra dataset
2. Candleize the main function of the model  
3. Build singularity container based on the curated model
4. TODO: Hyper Parameter Optimization (HPO)
## Requirements
- Create environment     
```
conda create -n TGSA python=3.7
```
  
- Install the necessary packages.
```
pip install git+https://github.com/ECP-CANDLE/candle_lib@develop   
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

pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-geometric==1.7.1
```

## Model running and testing
``` python
conda activate TGSA
git clone  https://github.com/JDACS4C-IMPROVE/TGSA.git
cd TGSA
chmod +x *.sh
######################
# Data Preprocessing
python pilot_preprocessing.py
# Model Training
python candle_train.py
# Model Testing
python test.py

######################
./train.sh 1 /tmp

```
## Build Singularity image
```
cd -
git clone https://github.com/JDACS4C-IMPROVE/Singularity.git
cd Singularity
rm config/improve.env
cp ./TGSA/singularity/improve.env ./Singularity/config/
./setup

rm definitions/*.def
cd -
cp ./TGSA/singularity/TGSA.def ./Singularity/definitions/

make   
make deploy 
```

## Run built Singularity container
```
# Download and prepreprocess data (optional, as the train.sh script will call this pre)
singularity exec --nv --bind /tmp:/candle_data_dir ./images/TGSA.sif /usr/local/TGSA/preprocessing.sh 4 /candle_data_dir
# Train the model
singularity exec --nv --bind /tmp:/candle_data_dir ./images/TGSA.sif /usr/local/TGSA/train.sh 4 /candle_data_dir
# Test the model
singularity exec --nv --bind /tmp:/candle_data_dir ./images/TGSA.sif /usr/local/TGSA/test.sh 
```
## Hyper Parameter Optimization (HPO)



