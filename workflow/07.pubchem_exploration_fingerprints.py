import os

# Data Manipulation
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm


# Rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

from rdkit.Chem import DataStructs


RAW_DATA_DIR = "../raw_data"

OUTPUT_DIR = "../data/07.pubchem_exploration"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():

    # Read complete data
    pchem_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "pubchem_random", "pubchem_100k_selected_smiles.tsv.gz"), sep='\t')


    # Convert all chemicals into fingerprints
    fps_dictionary = {cid:AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(pchem_df.loc[pchem_df["chem_id"]==cid, "smiles"].values[0]), 
                                                            radius=2,
                                                            nBits=1024) for cid in tqdm(pchem_df.chem_id)}
    
    # Save dictionary
    with open(os.path.join(OUTPUT_DIR, "fps_dictionary.pkl"), "wb") as file:
        pickle.dump(fps_dictionary, file)


if __name__ == "__main__":
    main()
    