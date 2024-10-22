import os
import yaml
import argparse
import torch
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from rdkit import Chem
from rdkit import RDLogger

from workflow.dataset.dataset_representation import batch_representation
from workflow.models.ginet_concat import GINet

RDLogger.DisableLog('rdApp.*') 



# Function to read command line arguments
def parse_arguments():

    """
    This function returns parsed command line arguments.

    """

    # Instantiate parser
    parser = argparse.ArgumentParser(prog="Represent molecular structures as using MolE.",
                                     description="This program recieves a file with SMILES and represents them using the MolE representation.",
                                     usage="python mole_representation.py smiles_filepath output_filepath [options]",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Input SMILES
    parser.add_argument("smiles_filepath", help="Complete path to the smiles filepath. Expects a TSV file with a column containing SMILES strings.")

    # Output filepath
    parser.add_argument("output_filepath", help="Complete path for the output.")

    # Column name for smiles
    parser.add_argument("-c", "--smiles_colname", help="Column name in smiles_filepath that contains the SMILES.",
                        default="smiles")
    
    # Column name for id
    parser.add_argument("-i", "--chemid_colname", help="Column name in smiles_filepath that contains the ID string of each chemical.",
                        default="chem_id")
    
    # MolE model
    parser.add_argument("-m", "--mole_model", help="Path to the directory containing the config.yaml and model.pth files of the pre-trained MolE chemical representation.",
                        default="pretrained_model/model_ginconcat_btwin_100k_d8000_l0.0001")
    
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


# A FUNCTION TO READ SMILES from file 
def read_smiles(data_path, smile_col="rdkit_no_salt", id_col="prestwick_ID"):

    """
    Read SMILES data from a file and remove invalid SMILES.

    Parameters:
    - data_path (str): Path to the file containing SMILES data.
    - smile_col (str, optional): Name of the column containing SMILES strings (default is "rdkit_no_salt").
    - id_col (str, optional): Name of the column containing molecule IDs (default is "prestwick_ID").

    Returns:
    - smile_df (pandas.DataFrame): DataFrame containing SMILES data with specified columns.
    """
    
    # Read the data
    smile_df = pd.read_csv(data_path, sep='\t')
    smile_df = smile_df[[smile_col, id_col]]

    # Make sure ID column is interpreted as str
    smile_df[id_col] = smile_df[id_col].astype(str)

    # Remove NaN
    smile_df = smile_df.dropna()

    # Remove invalid smiles
    smile_df = smile_df[smile_df[smile_col].apply(lambda x: Chem.MolFromSmiles(x) is not None)]

    return smile_df


# Function to load a pre-trained model
def load_pretrained_model(pretrained_model_dir, device="cuda:0"):

    """
    Load a pre-trained MolE model.

    Parameters:
    - pretrained_model_dir (str): Name of the pre-trained MolE model.
    - device (str, optional): Device for computation (default is "cuda:0").

    Returns:
    - model: Loaded pre-trained model.
    """

    # Read model configuration
    config = yaml.load(open(os.path.join(pretrained_model_dir, "config.yaml"), "r"), Loader=yaml.FullLoader)
    model_config = config["model"]

    # Instantiate model
    model = GINet(**model_config).to(device)
    
    # Load pre-trained weights
    model_pth_path = os.path.join(pretrained_model_dir, "model.pth")
    print(model_pth_path)

    state_dict = torch.load(model_pth_path, map_location=device)
    model.load_my_state_dict(state_dict)

    return model


def process_representation(dataset_path, smile_column_str, id_column_str, pretrained_dir, device):

    """
    Process the dataset to generate molecular representations.

    Parameters:
    - dataset_path (str): Path to the dataset file.
    - pretrained_dir (str): Name of the pre-trained model.
    - smile_column_str (str, optional): Name of the column containing SMILES strings.
    - id_column_str (str, optional): Name of the column containing molecule IDs.
    - device (str): Device to use for computation (default is "cuda:0"). Can also be "cpu".

    Returns:
    - udl_representation (pandas.DataFrame): DataFrame containing molecular representations if split_data=False.
    """

    # First we read the SMILES dataframe
    smiles_df = read_smiles(dataset_path, smile_col=smile_column_str, id_col=id_column_str)

    # Load the pre-trained model
    pmodel = load_pretrained_model(pretrained_model_dir=pretrained_dir, device=device)

    # Gather pre-trained representation
    udl_representation = batch_representation(smiles_df, pmodel, smile_column_str, id_column_str, device=device)

    return udl_representation


def main():

    # Parse arguments
    args = parse_arguments()

    # Obtain MolE pre-trained representation
    mole_representation = process_representation(dataset_path = args.smiles_filepath,
                                          smile_column_str = args.smiles_colname, 
                                          id_column_str = args.chemid_colname,

                                          
                                           pretrained_dir = args.mole_model,
                                           device=args.device)
    
    # Write MolE representation to output
    mole_representation.to_csv(args.output_filepath, sep='\t')


if __name__ == "__main__":
    main()
    
