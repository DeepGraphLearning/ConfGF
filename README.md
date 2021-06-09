![ConfGF](assets/logo.png)

----------------------------

[arXiv] | [Project Page] 

[arXiv]: https://arxiv.org/abs/2105.03902
[Project Page]: http://chenceshi.com/


The official implementation of Learning Gradient Fields for Molecular Conformation Generation (ICML 2021 **Long talk**)  

Code coming soon !

<p align="center">
  <img src="assets/sampling.png" /> 
</p>

<p align="center">
  <img src="assets/demo.gif" width="300">
</p>

## Installation

### Install via Conda (Recommended)


```bash
# Clone the environment
conda env create -f env.yml

# Activate the environment
conda activate confgf

# Install Library
git clone https://github.com/DeepGraphLearning/ConfGF.git
cd ConfGF
python setup.py install
```

### Install Manually

```bash
# Create conda environment
conda create -n confgf python=3.7

# Activate the environment
conda activate confgf

# Install packages
conda install -y -c pytorch pytorch=1.7.0 torchvision torchaudio cudatoolkit=10.2
conda install -y -c rdkit rdkit==2020.03.2.0
conda install -y scikit-learn pandas decorator ipython networkx tqdm matplotlib
conda install -y -c conda-forge easydict
pip install pyyaml

# Install PyTorch Geometric
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install torch-geometric==1.6.3

# Install Library
git clone https://github.com/DeepGraphLearning/ConfGF.git
cd ConfGF
python setup.py install
```


## Dataset 
### Offical Dataset
The offical raw dataset is avaiable here(https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF).
We also provide the preprocessed datasets in a dropbox folder(https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF).

### Prepare the raw dataset (optional)





## Citation
Please consider citing the following paper if you find our codes helpful. Thank you!
```
@inproceedings{shi*2021confgf,
title={Learning Gradient Fields for Molecular Conformation Generation},
author={Shi, Chence and Luo, Shitong and Xu, Minkai and Tang, Jian},
booktitle={International Conference on Machine Learning},
year={2021}
}
```

## Contact
Chence Shi (chence.shi@umontreal.ca)

