import os
import re
import argparse
import pickle
import torch
import numpy as np
import pandas as pd

from scipy.stats.mstats import gmean

from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

def parse_arguments():
    """
    This function returns parsed command line arguments.
    """

    # Instantiate parser
    parser = argparse.ArgumentParser(prog="Prediction of antimicrobial activity for peptide sequences",
                                     description="This program recieves a fasta file as input, featurizes the molecules using MolE, then makes predictions of antimicrobial activity",
                                     usage="python inference_script.py fasta_filepath outpath [options]")
    # Input FASTA
    parser.add_argument("input_filepath", help="Complete path to input FASTA file.")

    # Output file
    parser.add_argument("output_filepath", help="Complete path for output file")

    # Input type
    parser.add_argument("-s", "--smiles_input", help="Flag variable. Indicates if the input_filepath contains SMILES that have to be first represented using a MolE pre-trained model.",
                        action="store_true")
    
    # Column name for smiles
    parser.add_argument("-c", "--smiles_colname", help="Column name in input_filepath that contains the SMILES. Only used if smiles_input is set.",
                        default="smiles")
    
    # Column name for id
    parser.add_argument("-i", "--chemid_colname", help="Column name in smiles_filepath that contains the ID string of each chemical. Only used if smiles_input is set",
                        default="chem_id")
    

    # Indicate whether you want to aggregate the scores
    parser.add_argument("-a", "--aggregate_scores", help="Flag variable. If called, then prediction scores are aggregated by compound using as the antimicrobial potential of each compound.",
                        action="store_true")

    # XGBoost model
    parser.add_argument("-x", "--xgboost_model", help="Path to the pickled XGBoost model that makes predictions (.pkl). Default set to: data/04.new_predictions/MolE-XGBoost-08.03.2024_14.20.pkl",
                        default="data/03.model_evaluation/MolE-XGBoost-08.03.2024_14.20.pkl")

    # MolE model
    parser.add_argument("-m", "--mole_model", help="Path to the directory containing the config.yaml and model.pth files of the pre-trained MolE chemical representation. Only used if smiles_input is set. Default set to: mole_pretrained/model_ginconcat_btwin_100k_d8000_l0.0001",
                        default="pretrained_model/model_ginconcat_btwin_100k_d8000_l0.0001")

    # Maier Strains information
    parser.add_argument("-b", "--strain_categories", help="Path to the Maier et.al. screening results. Default is set to data/01.prepare_training_data/maier_screening_results.tsv.gz ",
                        default = "data/01.prepare_training_data/maier_screening_results.tsv.gz")
    
    # Additional information about the bacteria
    parser.add_argument("-g", "--gram_information", help="Path to strain metadata. Default is set to raw_data/maier_microbiome/strain_info_SF2.xlsx",
                        default = "raw_data/maier_microbiome/strain_info_SF2.xlsx")

    # Antimicrobial score threshold    
    parser.add_argument("-t", "--app_threshold", help="threshold score to binarize compound-microbe predictions. Default from original publication.",
                        default=0.04374140128493309)
    
    # Broad spectrum threshold
    parser.add_argument("-k", "--min_nkill", help="Minimum number of microbes predicted to be inhibited in order to consider broad spectrum antibiotic.",
                        default=10)

    # Device
    parser.add_argument("-d", "--device", help="Device where the pre-trained model is loaded. Can be one of ['cpu', 'cuda', 'auto']. If 'auto' (default) then cuda:0 device is selected if a GPU is detected.",
                        default="auto")

    # Parse arguments
    args = parser.parse_args()

    # Determine device for MolE model
    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"Using {args.device}")

    return args

def read_representation(args):

    """
    This function reads in the input file and returns the MolE representation
    """

    # Determine if input is smiles. If so, gather pre-trained representation
    if(args.smiles_input):
        # If input is SMILES, gather pre-trained representation
        from mole_representation import process_representation

        udl_df = process_representation(dataset_path=args.input_filepath,
                                                    smile_column_str=args.smiles_colname,
                                                    id_column_str=args.chemid_colname, 
                                                    
                                                    pretrained_dir=args.mole_model,
                                                    device=args.device)
        
    else:
        # If the input is already MolE representation, then read file
        udl_df = pd.read_csv(args.input_filepath, sep='\t', index_col=0)

    return udl_df


# Prepare the OHE of the strains
def prep_ohe(categories):

    """
    Prepare one-hot encoding for strain variables.

    This function creates a one-hot encoding representation of the provided categorical variables.
    It fits a OneHotEncoder to the categories and transforms them into a pandas DataFrame.

    Parameters:
    - categories (array-like): Array-like object containing categorical variables.

    Returns:
    - cat_ohe (pandas.DataFrame): DataFrame representing the one-hot encoded categorical variables.
    """

    # Prepare OHE
    ohe = OneHotEncoder(sparse=False)

    # Fit OHE
    ohe.fit(pd.DataFrame(categories))

    # Prepare OHE
    cat_ohe = pd.DataFrame(ohe.transform(pd.DataFrame(categories)), columns=categories, index=categories)

    return cat_ohe

def add_strains(chemfeats_df, screen_path):

    """
    Add strains to chemical features using Cartesian product merge.

    This function adds strains to chemical features using Cartesian product merge
    between the chemical features DataFrame and the one-hot encoded strains DataFrame.

    Parameters:
    - chemfeats_df (pandas.DataFrame): DataFrame containing chemical features.
    - screen_path (str): Path to the maier_screening_results.tsv file.

    Returns:
    - xpred (pandas.DataFrame): DataFrame containing chemical features with added strains.
    """

    # Read screen information and One-hot-encode strains
    maier_screen = pd.read_csv(screen_path, sep='\t', index_col=0)
    ohe_df = prep_ohe(maier_screen.columns)


     # Prepare chemical features
    chemfe = chemfeats_df.reset_index().rename(columns={"index": "chem_id"})
    chemfe["chem_id"] = chemfe["chem_id"].astype(str) 

    # Prepare OHE
    sohe = ohe_df.reset_index().rename(columns={"index": "strain_name"})

    # Cartesian product merge
    xpred = chemfe.merge(sohe, how="cross")
    xpred["pred_id"] = xpred["chem_id"].str.cat(xpred["strain_name"], sep=":")

    xpred = xpred.set_index("pred_id")
    xpred = xpred.drop(columns=["chem_id", "strain_name"])

    # Make sure correct number of rows
    assert xpred.shape[0] == (chemfeats_df.shape[0] * ohe_df.shape[0])

    # Make sure correct number of features
    assert xpred.shape[1] == (chemfeats_df.shape[1] + ohe_df.shape[1])
    
    return xpred

def load_xgb_model(xgb_path):
    """
    This function loads an XGBoost model that makes predictions of antimcrobial activity

    Parameters
    - xgb_path (str): Path to a pickled XGBoost model

    Returns:
    - model (XGBClassifier): An XGBClassifier object
    """

    with open(xgb_path, "rb") as file:
        model = pickle.load(file)

    return model

def gram_stain(label_df, strain_info_df):

    """
    Add Gram stain information to strain labels.

    This function adds Gram stain information to strain labels based on the NT number in the strain name.

    Parameters:
    - label_df (pandas.DataFrame): DataFrame containing strain labels.
    - strain_info_df (pandas.DataFrame, optional): DataFrame containing strain information (default is maier_strains).

    Returns:
    - df_label (pandas.DataFrame): DataFrame containing strain labels with Gram stain information.
    """

    # Create copy of the label dataframe
    df_label = label_df.copy()
    
    # Gather NT number
    df_label["nt_number"] = df_label["strain_name"].apply(lambda x: re.search(".*?\((NT\d+)\)", x).group(1))

    # Create Gram strain dict
    gram_dict = strain_info_df[["Gram stain"]].to_dict()["Gram stain"]

    # Add stain information
    df_label["gram_stain"] = df_label["nt_number"].apply(gram_dict.get)

    return df_label

def antimicrobial_potential(score_df, strain_filepath):

    # Read the metadata of strains.
    maier_strains = pd.read_excel(strain_filepath,
                             skiprows=[0,1, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54], 
                             index_col="NT data base")

    # Separate chem_id from strain name
    score_df["chem_id"] = score_df["pred_id"].str.split(":", expand=True)[0]
    score_df["strain_name"] = score_df["pred_id"].str.split(":", expand=True)[1]

    # Add gram stain information
    pred_df = gram_stain(score_df, maier_strains)


    # Aggregate complete antimicrobial potential
    # Antimicrobial score
    apscore_total = pred_df.groupby("chem_id")["1"].apply(gmean).to_frame().rename(columns={"1": "apscore_total"})

    # Antimicrobial score by gram stain
    apscore_gram = pred_df.groupby(["chem_id", "gram_stain"])["1"].apply(gmean).unstack().rename(columns={"negative": "apscore_gnegative",
                                                                                                         "positive": "apscore_gpositive"})
    
    # Number inhibited strains
    inhibted_total = pred_df.groupby("chem_id")["growth_inhibition"].sum().to_frame().rename(columns={"growth_inhibition": "ginhib_total"})

    # Number inhibited strains per stain
    inhibted_gram = pred_df.groupby(["chem_id", "gram_stain"])["growth_inhibition"].sum().unstack().rename(columns={"negative": "ginhib_gnegative",
                                                                                                                    "positive": "ginhib_gpositive"})
    # Merge the results
    agg_pred = apscore_total.join(apscore_gram).join(inhibted_total).join(inhibted_gram)

    return agg_pred


def main():

    # Read input arguments
    args = parse_arguments()

    # Determine if input is smiles. If so, gather pre-trained representation
    udl_representation = read_representation(args)

    # Prepare strain-level predictions
    X_input = add_strains(udl_representation, args.strain_categories)

    # Read XGBoost model
    model_abx = load_xgb_model(args.xgboost_model)

    # Make predictions
    y_pred = model_abx.predict_proba(X_input)
    pred_df = pd.DataFrame(y_pred, columns = ["0", "1"], index=X_input.index)

    # Binarize predictions using threshold
    pred_df["growth_inhibition"] = pred_df["1"].apply(lambda x: 1 if x >= args.app_threshold else 0)

    # Determine if results should be aggregated
    if(args.aggregate_scores):
        print("Aggregating Antimicrobial potential")

        pred_df = pred_df.reset_index()

        agg_df = antimicrobial_potential(pred_df, args.gram_information)

        # Determine if chemical is broad spectrum
        agg_df["broad_spectrum"] = agg_df["ginhib_total"].apply(lambda x: 1 if x >= args.min_nkill else 0)

        # Write file
        agg_df.to_csv(args.output_filepath, sep='\t')
    
    else:
        pred_df.to_csv(args.output_filepath, sep='\t')


if __name__ == "__main__":
    main()