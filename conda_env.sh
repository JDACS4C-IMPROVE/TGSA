#!/bin/bash --login

set -e

# This script was create with the help from here:
# https://www.youtube.com/watch?v=lu2DzaqBeDg
# https://github.com/kaust-rccl/conda-environment-examples/tree/pytorch-geometric-and-friends


# From the original repo itself
# conda create -n TGSA python=3.6 pip
conda install -c rdkit rdkit
pip install fitlog
pip install torch==1.6.0
pip install torch-cluster==1.5.9
pip install torch-scatter==2.0.6
pip install torch-sparse==0.6.9
pip install torch-spline-conv==1.2.1
pip install torch-geometric==1.6.1
conda install -c dglteam dgllife

# Check
# python -c "import torch; print(torch.__version__)"
# python -c "import torch; print(torch.version.cuda)"
# python -c "import torch_geometric; print(torch_geometric.__version__)"

# My packages
conda install -c conda-forge ipdb=0.13.9 --yes
conda install -c conda-forge jupyterlab=3.2.0 --yes
conda install -c conda-forge python-lsp-server=1.2.4 --yes
