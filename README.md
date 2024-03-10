# Using MolE to predict antimicrobial activity  

In this project, we use MolE's pre-trained representation to train XGBoost models to predict the antimicrobial activity of compounds based on their molecular structure.  

In the `workflow` directory you will find the steps followed for our analysis as a collection of ordered Jupyter Notebooks. A general overview of the steps followed can be found in the same directory.

The `data` directory contains the output of each step in `workflow` in a subfolder with the same name as the corresponding Notebook.   

The `pretrained_model` directory contains the MolE pre-trained model used to get the molecular representation.  

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

## Datasets
This section is a brief description of the datasets used for our study. All raw data can be found in the `raw_data` directory.

### Maier-2018 Library

The data used in this study was obtained from the Supplementary Information of [Maier, 2018](https://www.nature.com/articles/nature25979#Abs1). The data can be found in `raw_data/maier_microbiome` subdirectory. In there, you will find:  

1. chem_library_info_SF1.xlsx: An overview of the 2,000 compounds that make up the Prestwick Library used the original study.
2. screen_results_info_SF3.xlsx: The adjusted p-value table the results from the screening of 1,197 compounds against 40 bacterial strains.
3. strain_info_SF2.xlsxP Additional information about the strains used in the original study.

### MedChemExpress Library
In the `raw_data/medchemexpress` subdirectory you will find the `chemical_library_cleaned.tsv.gz` file. This file contains information gathered directly from the MedChemExpress provider. This includes information such as the `Catalog Number` and the `Biological Activity` descriptions. In addition, in this file the known antibiotics are already highlighted as well as the corresponding ATC code.


## Models  
Here you can find some details of the pre-trained model for molecular representation and the XGBoost model used for antimicrobial prediction.

### Pre-trained MolE
Pre-training details can be found in our manuscript and the MolE repository. The hyperparameters of the pre-trained model are specified in `pretrained_model/model_ginconcat_btwin_100k_d8000_l0.0001/config.yaml`. Briefly, they are the following: $\lambda = 10^{-4} \text{,  } z \in \mathbb{R}^{8000}$, pre-trained on 100,000 molecules.   

You can download the model binary file from Zenodo and place it in the `pretrained_model/model_ginconcat_btwin_100k_d8000_l0.0001/` subdirectory.  
To get the pre-trained representation of a new set of molecules you will need a tab-separated file that contains:  

1. A column that identifies each molecule
2. A column that contains the SMILES of each molecule (preferably canonical SMILES with salts removed)

In `workflow/04.new_predictions.ipynb` you can find examples of how to obtain the pre-trained representation using the `process_dataset()` function. Here is a snippet on how we obtain the representation for [Halicin](https://www.sciencedirect.com/science/article/pii/S0092867420301021) and [Abaucin](https://www.nature.com/articles/s41589-023-01349-8)

```
# Import the relevant function
from dataset.dataset_representation import process_dataset

# Here we obtain the representation using the relevant function.
mole_representation = process_dataset(dataset_path = "novel_abx_smiles.tsv.gz", 
                                                  pretrain_architecture = "gin_concat", 
                                                  pretrained_model = "model_ginconcat_btwin_100k_d8000_l0.0001", 
                                                  split_data=False,
                                                  smile_column_str = "pchem_canonical_smiles", 
                                                  id_column_str = "compound")
```

The same function can also be used to split the data into training, validation, and testing sets following the scaffold splitting procedure. Examples of this can be seen in `workflow/01.prepare_training_data.ipynb`

```
scaffold_split, mole_representation = process_dataset(dataset_path = "prestwick_library_screened.tsv.gz", 
                                                  pretrain_architecture = "gin_concat", 
                                                  pretrained_model = "model_ginconcat_btwin_100k_d8000_l0.0001", 
                                                  split_approach = "scaffold", 
                                                  validation_proportion = 0.1, 
                                                  test_proportion = 0.1, 
                                                  smile_column_str = "rdkit_no_salt", 
                                                  id_column_str = "prestwick_ID")
```
Further documentation for this function can be found in the docstrings `workflow/dataset/dataset_representation.py`

### XGBoost model

The binary files of the XGBoost models trained to predict antimicrobial activity can be found in `data/03.model_evaluation`. The model can be loaded using the [pickle](https://docs.python.org/3/library/pickle.html) module. 

```
# Import pickle module
import pickle

# Load the model
with open("MolE-XGBoost-08.03.2024_14.20.pkl", "rb") as file:
    mole_model = pickle.load(file)
```

Predictions of activity can be made for each strain present in the [Maier, 2018](https://www.nature.com/articles/nature25979#Abs1) study. This is done by concatenating one-hot-encoded vectors to the molecular representation. The resulting matrix will have $N_{molecules} * N_{strains}$ rows. In this way, predictions can be made for each compound-strain combination. Examples of how this is done can be seen in `workflow/04.new_predictions.ipynb` using the `add_strains()` function.  

```
# Prepare the One-hot-encoding for the strains
strain_ohe = prep_ohe(strain_categories)

# Prepare the input matrix. 
x_input = add_strains(mole_representation, strain_ohe)

# Make predictions
y_scores = mole_model.predict_proba(x_input)
```

To binarize the resulting scores into a `1` or `0` label, one can use the optimized score thresholds. The optimized thresholds can be found in `data/03.model_evaluation/optimal_thresholds.tsv.gz`.

```
# Read in the optimal threshold data.
threhold_df = pd.read_csv("optimal_thresholds.tsv.gz", sep='\t')

# Isolate the value to be used as the optimal threshold for the MolE model.
mole_optimal_threshold = threhold_df.loc[(threhold_df["representation"]== "MolE") & (threhold_df["score_type"]== "optimized"), "threshold"].values[0]

# Apply the threshold to binarize the labels
pred_df = pd.DataFrame(y_scores, columns=["pred_score"], index=x_input.index)

# Apply the threshold
pred_df["pred_label"] = pred_df["pred_score"].apply(lambda x: 1 if x >= mole_optimal_threshold else 0)
```

As a result, one has the predicted outcome for each compound-strain combination. Further downstream analysis can be found in `workflow/04.new_predictions.ipynb`


