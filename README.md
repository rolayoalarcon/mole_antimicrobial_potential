# Using MolE to predict antimicrobial activity  

In this project, we use MolE's pre-trained representation to train XGBoost models to predict the antimicrobial activity of compounds based on their molecular structure.  

In the `workflow` directory you will find the steps followed for our analysis as a collection of ordered Jupyter Notebooks. A general overview of the steps followed can be found in the same directory.

The `data` directory contains the output of each step in `workflow` in a subfolder with the same name as the corresponding Notebook.   

The `pretrained_model` directory contains the MolE pre-trained model that is used to get the molecular representation.  

Finally, the `raw_data` directory contains files from external references. This includes the data from [Maier, 2018](https://www.nature.com/articles/nature25979#Abs1), as well as the chemical library on which we make new predictions. 

## Environment setup  
You should be able to reproduce the main findings by creating a conda environment.  

```
# Create a conda environment
$ conda create --name mole python=3.7
$ conda activate mole

# Install the other packages used
# Install pytorch
$ pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install torch-geometric==1.6.3 torch-sparse==0.6.9 torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html

# Install additional dependencies
$ pip install PyYAML==6.0 pandas==1.3.5 numpy==1.21.6 xgboost==1.6.2 scikit_learn==1.0.2 seaborn==0.11.2

# Install RDKit
$ conda install -c conda-forge rdkit=2020.09.1.0
```

## Models  
Here you can find some details of the pre-trained model for molecular representation, as well as the XGBoost model used for antimicrobial prediction.

### Pre-trained MolE
Pre-training details can be found in our manuscript, and in the MolE repository. The hyperparameters of the pre-trained model are specified in `pretrained_model/model_ginconcat_btwin_100k_d8000_l0.0001/config.yaml`. Briefly, they are the following: $\lambda = 10^{-4} \text{,  } z \in \mathbb{R}^{8000}$, pre-trained on 100,000 molecules.   

You can download the model binary file from Zenodo and place it in `pretrained_model/model_ginconcat_btwin_100k_d8000_l0.0001/` subdirectory.  
To get the pre-trained representation of a new set of molecules you will need a tab-separated file that contains:  

1. A column that identifies each molecule
2. A column that contains the SMILES of each molecule (preferably canonical SMILES with salts removed)





