# Using MolE to predict antimicrobial activity  

In this project, we use MolE's pre-trained representation to train XGBoost models to predict the antimicrobial activity of compounds based on their molecular structure. This repository contains all of the scripts and data used to obtain the results presented in our [paper](#publication).  

For more information about the MolE pre-training framework, you can visit the [MolE](https://github.com/rolayoalarcon/MolE) repository.

## Publication
For more information about MolE, and how we use it to predict antimicrobial activity, you can check out our paper in Nature Communications:  
[**Pre-trained molecular representations enable antimicrobial discovery**](https://www.nature.com/articles/s41467-025-58804-4)  

## Contents  
This repository contains all of the scripts and data used to obtain the results presented in our [pre-print](#pre-print).   
You will find the following directories:  
- `workflow`: Contains the steps followed for our analysis as a collection of ordered Jupyter Notebooks. You will also find some `.Rmd` files, though these are mainly for plotting. An overview of the steps followed can be found in the README of the subdirectory.
- `data`: Contains the output of each step in `workflow` in a subfolder with the same name as the corresponding Notebook.  
- `pretrained_model`: Contains the MolE pre-trained model used to get the molecular representation used to train our model for antimicrobial prediction.  
- `raw_data`: Contains files from external references. This includes the training data from [Maier, 2018](https://www.nature.com/articles/nature25979), the MCE chemical library, results from experimental validation, and a random selection of 100K compounds from PubChem.  

Additionally, we provide two scripts that can help you perfom a quick analysis on your own set of molecules:  
- [mole_representation.py](#obtaining-molecular-representations): Gather the pre-trained representation of a collection of compounds.
- [mole_antimicrobial_prediction.py](#predicting-antimicrobial-activity): Predict the antimicrobial activity of a collection of compounds.

More details for each script can be found below.

## Environment setup  
For convinience, you can create a conda environment with all the necessary dependencies. To make things easier, we provide an `environment.yaml` file, that you can use to set up the environment with all the necessary dependencies. Keep in mind, that we install [pytorch assuming a CUDA 11.8 compute platform](https://pytorch.org/).

```
# Create the conda environment with name mole
conda env create -f environment.yaml

# Afterwards activate the environment
conda activate mole
```

Once this is done, you can clone this repository.
  
## Models  
In this project, we used two kinds of models: 1. the pre-trained MolE model that represents molecular structures, and 2. XGBoost models that receive molecular representations and predict antimicrobial activity. Here you can find more details about both.  

### Pre-trained MolE model  
The hyperparameters of the pre-trained model used in this work are specified in `pretrained_model/model_ginconcat_btwin_100k_d8000_l0.0001/config.yaml`.

You can download the model binary file from [here](https://zenodo.org/records/10803099?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImI3NTg0OTU0LTI5YWItNDgxZS04OGYyLTU5MmM1MjcwYzJjZiIsImRhdGEiOnt9LCJyYW5kb20iOiIzNzgyNTE5ZGU5N2MzZWI3YjZiZjkwYTIzZjFiMmEwZSJ9.oL6G0WZKxIowSb-2qdP55cPhef1W4yG5iF4PFlsWPpuPROmzRhutJtySzs9q02ACltl0qy9YPJjzB7NvzRMyaw) and place the `model.pth` file in the `pretrained_model/model_ginconcat_btwin_100k_d8000_l0.0001/` subdirectory.  
  
**IMPORTANT:** You will need to download the model binary in order to use the prepared Notebooks in `workflow` and the provided scripts.  
  
### XGBoost model for antimicrobial prediction  

You will also find binary files for the XGBoost models that were trained to predict the antimicrobial activity of compounds in `data/03.model_evaluation` as `*.pkl` files. The model that uses MolE representations is called `MolE-XGBoost-08.03.2024_14.20.pkl`.
  
## Obtaining molecular representations  

You can obtain MolE's pre-trained representation for your own set of molecules using the `mole_representation.py` script.

```
$ python mole_representation.py -h

usage: python mole_representation.py smiles_filepath output_filepath [options]

This program recieves a file with SMILES and represents them using the MolE representation.

positional arguments:
  smiles_filepath       Complete path to the smiles filepath. Expects a TSV file with a column containing SMILES strings.
  output_filepath       Complete path for the output.

optional arguments:
  -h, --help            show this help message and exit
  -c SMILES_COLNAME, --smiles_colname SMILES_COLNAME
                        Column name in smiles_filepath that contains the SMILES. (default: smiles)
  -i CHEMID_COLNAME, --chemid_colname CHEMID_COLNAME
                        Column name in smiles_filepath that contains the ID string of each chemical. (default: chem_id)
  -m MOLE_MODEL, --mole_model MOLE_MODEL
                        Path to the directory containing the config.yaml and model.pth files of the pre-trained MolE chemical representation. Default set to:
                        mole_pretrained/model_ginconcat_btwin_100k_d8000_l0.0001 (default: pretrained_model/model_ginconcat_btwin_100k_d8000_l0.0001)
  -d DEVICE, --device DEVICE
                        Device where the pre-trained model is loaded. Can be one of ['cpu', 'cuda', 'auto']. If 'auto' (default) then cuda:0 device is selected if a GPU is
                        detected. (default: auto)
```
  
An example of the type of input expected can be found in `examples/input/examples_molecules.tsv`. Using this script, you can obtain the MolE representation for each molecule with the following command.   
```
$ python mole_representation.py examples/input/examples_molecules.tsv examples/output/example_molecules_mole.tsv.gz --chemid_colname chem_name
```
  
The corresponding output can be found in `examples/output/example_molecules_mole.tsv.gz`.  

## Predicting antimicrobial activity  

You can make predictions of the antimicrobial activity for your collection of molecules using the `mole_antimicrobial_prediction.py` script. The activity of each compound is predicted for each bacterial strain present in the [Maier, 2018](https://www.nature.com/articles/nature25979) study. Predictions can also be aggregated for each compound in order to identify broad-spectrum antimicrobials.  
  
```
$ python mole_antimicrobial_prediction.py -h 

usage: python mole_antimicrobial_prediction.py input_filepath output_filepath [options]

This program recieves a collection of molecules as input. If it receives SMILES, it first featurizes the molecules using MolE, then makes predictions of antimicrobial activity. By default, the program returns the
antimicrobial predictive probabilities for each compound-strain pair. If the --aggregate_scores flag is set, then the program aggregates the predictions into an antimicrobial potential score and reports the number of strains
inhibited by each compound.

positional arguments:
  input_filepath        Complete path to input file. Can be a file with SMILES (make sure to set the --smiles_input flag) or a file with MolE representation.
  output_filepath       Complete path for output file

optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
                        Device where the pre-trained model is loaded. Can be one of ['cpu', 'cuda', 'auto']. If 'auto' (default) then cuda:0 device is selected if a GPU is detected. (default: auto)

Input arguments:
  Arguments related to the input file

  -s, --smiles_input    Flag variable. Indicates if the input_filepath contains SMILES that have to be first represented using a MolE pre-trained model. (default: False)
  -c SMILES_COLNAME, --smiles_colname SMILES_COLNAME
                        Column name in input_filepath that contains the SMILES. Only used if --smiles_input is set. (default: smiles)
  -i CHEMID_COLNAME, --chemid_colname CHEMID_COLNAME
                        Column name in smiles_filepath that contains the ID string of each chemical. Only used if --smiles_input is set (default: chem_id)

Model arguments:
  Arguments related to the models used for prediction

  -x XGBOOST_MODEL, --xgboost_model XGBOOST_MODEL
                        Path to the pickled XGBoost model that makes predictions (.pkl). (default: data/03.model_evaluation/MolE-XGBoost-08.03.2024_14.20.pkl)
  -m MOLE_MODEL, --mole_model MOLE_MODEL
                        Path to the directory containing the config.yaml and model.pth files of the pre-trained MolE chemical representation. Only used if smiles_input is set. (default:
                        pretrained_model/model_ginconcat_btwin_100k_d8000_l0.0001)

Prediction arguments:
  Arguments related to the prediction process.

  -a, --aggregate_scores
                        Flag variable. If not set, then the prediction for each compound-strain pair is reported. If set, then prediction scores of each compound is aggregated into the antimicrobial potential score and the
                        total number of strains predicted to be inhibited is reported. Additionally, the broad spectrum antibiotic prediction is reported. (default: False)
  -t APP_THRESHOLD, --app_threshold APP_THRESHOLD
                        Threshold score applied to the antimicrobial predictive probabilities in order to binarize compound-microbe predictions of growth inhibition. Default from original publication. (default:
                        0.04374140128493309)
  -k MIN_NKILL, --min_nkill MIN_NKILL
                        Minimum number of microbes predicted to be inhibited in order to consider the compound a broad spectrum antibiotic. (default: 10)

Metadata arguments:
  Arguments related to the metadata used for prediction.

  -b STRAIN_CATEGORIES, --strain_categories STRAIN_CATEGORIES
                        Path to the Maier et.al. screening results. (default: data/01.prepare_training_data/maier_screening_results.tsv.gz)
  -g GRAM_INFORMATION, --gram_information GRAM_INFORMATION
                        Path to strain metadata. (default: raw_data/maier_microbiome/strain_info_SF2.xlsx)
```

You can pass a file with SMILES as input and the program will first obtain the MolE representation in order to make predictions. In this case, make sure to set the `--smiles_input` flag argument.  

```
$ python mole_antimicrobial_prediction.py examples/input/examples_molecules.tsv examples/output/example_molecules_prediction.tsv --smiles_input --chemid_colname chem_name

```
  
Alternatively, if you have already obtained the MolE representation of your molecules (for example, using [mole_representation.py](#obtaining-molecular-representations)) you can use this directly.
  
```
$ python mole_antimicrobial_prediction.py examples/output/example_molecules_mole.tsv.gz examples/output/example_molecules_prediction.tsv
```
  
By default, the script will return predictions for each compound-microbe pair. The complete output for the examples above is in `examples/output/example_molecules_prediction.tsv`. A preview of that file can be seen below:
  
| pred_id |	antimicrobial_predictive_probability | growth_inhibition |
| ------- | ------------------------------------ | ------------------ |
| Halicin:Akkermansia muciniphila (NT5021) | 0.021192694 | 0 |
| Halicin:Bacteroides caccae (NT5050) |	0.20044225 | 1 |
| Halicin:Bacteroides fragilis (ET) (NT5033) | 0.13638856 | 1 |

- **pred_id**: Indicates the `compound : microbe` combination for which the prediction is made.
- **antimicrobial_predictive_probability**: Indicates the prediction score for the compound inhibiting the growth of the microbe.
- **growth_inhibition**: A binary column indicating whether the compound is predicted to inhibit the microbe's growth (1) or not (0). This is acheived by thresholding the **antimicrobial_predictive_probability** values, using a pre-determined score threshold. By default, the same threshold used in our publication is applied.
  

Additionally, these pairwise predictions can be aggregated in order to identify predicted broad-spectrum antimicrobials and obtain the antimicrobial_potential_score as described in our [pre-print](#pre-print). To do this, one can simply set the `--aggregate_scores` flag. 
  
```
$ python mole_antimicrobial_prediction.py examples/input/examples_molecules.tsv examples/output/example_molecules_prediction_aggregated.tsv --smiles_input --chemid_colname chem_name --aggregate_scores
```
 
The complete output can be found in `examples/output/example_molecules_prediction_aggregated.tsv`. Below is a preview:  
  
| chem_id | apscore_total |	apscore_gnegative |	apscore_gpositive |	ginhib_total | ginhib_gnegative | ginhib_gpositive | broad_spectrum |
| ------- | ------------- | ----------------- | ----------------- | ------------ | ---------------- | ---------------- | -------------- |
| Halicin | -1.8453829 | -1.8742418 | -1.8217711 | 35 | 16 | 19 |	1
| Abaucin |	-7.775602 |	-8.003299 |	-7.589305 |	1 |	0 |	1 |	0 |
| Diacerein |	-2.6554723 |	-3.4372644 |	-2.0158243 |	26 |	9 |	17 |	1 |
  
- **chem_id**: The identifier of each compount.
- **apscore_total**: Is the aggregated Antimicrobial Potential score of each compound.
- **apscore_gnegative**: Is the aggregated Antimicrobial Potential score of each compound on the subset of Gram Negative species.
- **apscore_gpositive**: Is the aggregated Antimicrobial Potential score of each compound on the subset of Gram Positive species.
- **ginhib_total**: Is the total amount strains that are predicted to be inhibited by each compound.
- **ginhib_gpositive**: Is the total amount Gram Positive strains that are predicted to be inhibited by each compound.
- **ginhib_gnegative**: Is the total amount Gram Negative strains that are predicted to be inhibited by each compound.
- **broad_spectrum**: Is a binary column indicating whether the compound is predicted to be a broad spectrum antimicrobial (1) or not (0). This is done by thresholding the total amount of strains predicted to be inhibited (**ginhib_total**). By default if a compound is predicted to inhibit $\geq$ 10 strains, then it is predicted to have broad spectrum activity.
