#!/bin/bash --login

set -e

# This script was create with the help from here:
# https://www.youtube.com/watch?v=lu2DzaqBeDg
# https://github.com/kaust-rccl/conda-environment-examples/tree/pytorch-geometric-and-friends


# Manually run these commands before running this sciprt
# conda env create --file environment.yml --force
# conda activate py36tgsa
# Run script as follows:
# $ ./conda_create_on_lamina.sh
python -m pip install torch-scatter --no-cache-dir --no-index --find-links https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
python -m pip install torch-sparse --no-cache-dir --no-index --find-links https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
python -m pip install torch-cluster --no-cache-dir --no-index --find-links https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
python -m pip install torch-spline-conv --no-cache-dir --no-index --find-links https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
python -m pip install torch-geometric --no-cache-dir
# python -m pip install torch-geometric==1.6.1 --no-cache-dir

conda install -c rdkit rdkit --yes
python -m pip install fitlog==0.9.13

# My packages
conda install -c conda-forge ipdb=0.13.9 --yes
conda install -c conda-forge jupyterlab=3.2.0 --yes
conda install -c conda-forge python-lsp-server=1.2.4 --yes

# Check
# python -c "import torch; print(torch.__version__)"
# python -c "import torch; print(torch.version.cuda)"
# python -c "import torch_geometric; print(torch_geometric.__version__)"

# # creates the conda environment
# PROJECT_DIR=$PWD
# conda env create --prefix $PROJECT_DIR/env --file $PROJECT_DIR/environment.yml --force

# # activate the conda env before installing PyTorch Geometric via pip
# conda activate $PROJECT_DIR/env
# TORCH=1.6.0
# CUDA=cu102
# python -m pip install torch-scatter --no-cache-dir --no-index --find-links https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
# python -m pip install torch-sparse --no-cache-dir --no-index --find-links https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
# python -m pip install torch-cluster --no-cache-dir --no-index --find-links https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
# python -m pip install torch-spline-conv --no-cache-dir --no-index --find-links https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
# python -m pip install torch-geometric --no-cache-dir
