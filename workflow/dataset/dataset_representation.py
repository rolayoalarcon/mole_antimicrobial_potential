import os
import yaml
import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Data, Dataset, Batch

from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit import RDLogger                           
from rdkit.Chem.SaltRemover import SaltRemover    

from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import MolSurf
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors3D

RDLogger.DisableLog('rdApp.*')  


ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

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
    smile_df = pd.read_csv(data_path, sep='\t', index_col=0)
    smile_df = smile_df[[smile_col, id_col]]

    # Remove NaN
    smile_df = smile_df.dropna()

    # Remove invalid smiles
    smile_df = smile_df[smile_df[smile_col].apply(lambda x: Chem.MolFromSmiles(x) is not None)]

    return smile_df

# A FUNCTION TO CALCULATE DESCRIPTORS
def calc_descriptors(smiles, id):
    """
    Calculate molecular descriptors for a given SMILES string.
    ORIGINAL FUNCTION FROM ALGAVI & BORENSTEIN, 2023

    Parameters:
    - smiles (str): SMILES string of the molecule.
    - id (str): ID of the molecule.

    Returns:
    - smiles_df (pandas.DataFrame): DataFrame containing calculated molecular descriptors.
    """

    #define mol
    mol = Chem.MolFromSmiles(smiles)
    mol1=Chem.AddHs(mol)
    
    #remove salts
    remover = SaltRemover()
    mol1 =  remover.StripMol(mol1)
    smiles_df = pd.DataFrame({"chem_id": [id]})
    
    #calculate conformers
    AllChem.EmbedMolecule(mol1)
    AllChem.MMFFOptimizeMolecule(mol1)
    
    ## genral stracture descriptors (rdkit.Chem.Descriptors module)
    smiles_df["MolWt"] = Chem.Descriptors.MolWt(mol1)
    smiles_df["BertzCT"] = Chem.Descriptors.BertzCT(mol1)
    smiles_df["MolLogP"] = Chem.Descriptors.MolLogP(mol1)
    smiles_df["MolMR"] = Chem.Descriptors.MolMR(mol1)
    smiles_df["HeavyAtomCount"] = Chem.Descriptors.HeavyAtomCount(mol1)
    smiles_df["NumHAcceptors"] = Chem.Descriptors.NumHAcceptors(mol1)
    smiles_df["NumHDonors"] = Chem.Descriptors.NumHDonors(mol1)
    smiles_df["NumValenceElectrons"] = Chem.Descriptors.NumValenceElectrons(mol1)
    smiles_df["RingCount"] = Chem.Descriptors.RingCount(mol1)
    smiles_df["FractionCSP3"] = Chem.Descriptors.FractionCSP3(mol1)
    smiles_df["NHOHCount"] = Chem.Descriptors.NHOHCount(mol1)
    smiles_df["NOCount"] = Chem.Descriptors.NOCount(mol1)
    smiles_df["HeavyAtomMolWt"] = Chem.Descriptors.HeavyAtomMolWt(mol1)
    smiles_df["MaxAbsPartialCharge"] = Chem.Descriptors.MaxAbsPartialCharge(mol1)
    smiles_df["MaxPartialCharge"] = Chem.Descriptors.MaxPartialCharge(mol1)
    smiles_df["MinAbsPartialCharge"] = Chem.Descriptors.MinAbsPartialCharge(mol1)
    smiles_df["MinPartialCharge"] = Chem.Descriptors.MinPartialCharge(mol1)

    #Graph descriptors from Chem.rdMolDescriptors module
    smiles_df["Chi0n"] =Chem.rdMolDescriptors.CalcChi0n(mol1)
    smiles_df["Chi0v"] =Chem.rdMolDescriptors.CalcChi0v(mol1)
    smiles_df["Chi1n"] =Chem.rdMolDescriptors.CalcChi1n(mol1)
    smiles_df["Chi1v"] =Chem.rdMolDescriptors.CalcChi1v(mol1)
    smiles_df["Chi2n"] =Chem.rdMolDescriptors.CalcChi2n(mol1)
    smiles_df["Chi2v"] =Chem.rdMolDescriptors.CalcChi2v(mol1)
    smiles_df["Chi3n"] =Chem.rdMolDescriptors.CalcChi3n(mol1)
    smiles_df["Chi3v"] =Chem.rdMolDescriptors.CalcChi3v(mol1)
    smiles_df["Chi4n"] =Chem.rdMolDescriptors.CalcChi4n(mol1)
    smiles_df["Chi4v"] =Chem.rdMolDescriptors.CalcChi4v(mol1)
    smiles_df["HallKierAlpha"] =Chem.rdMolDescriptors.CalcHallKierAlpha(mol1)
    smiles_df["Kappa1"] =Chem.rdMolDescriptors.CalcKappa1(mol1)
    smiles_df["Kappa2"] =Chem.rdMolDescriptors.CalcKappa2(mol1)
    smiles_df["Kappa3"] =Chem.rdMolDescriptors.CalcKappa3(mol1)
    smiles_df["LabuteASA"] =Chem.rdMolDescriptors.CalcLabuteASA(mol1)
    smiles_df["NumAliphaticCarbocycles"] =Chem.rdMolDescriptors.CalcNumAliphaticCarbocycles(mol1)
    smiles_df["NumAliphaticHeterocycles"] =Chem.rdMolDescriptors.CalcNumAliphaticHeterocycles(mol1)
    smiles_df["NumAliphaticRings"] =Chem.rdMolDescriptors.CalcNumAliphaticRings(mol1)
    smiles_df["NumAmideBonds"] =Chem.rdMolDescriptors.CalcNumAmideBonds(mol1)
    smiles_df["NumAromaticCarbocycles"] =Chem.rdMolDescriptors.CalcNumAromaticCarbocycles(mol1)
    smiles_df["NumAromaticHeterocycles"] =Chem.rdMolDescriptors.CalcNumAromaticHeterocycles(mol1)
    smiles_df["NumAromaticRings"] =Chem.rdMolDescriptors.CalcNumAromaticRings(mol1)
    smiles_df["NumBridgeheadAtoms"] =Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(mol1)
    smiles_df["NumHBA"] =Chem.rdMolDescriptors.CalcNumHBA(mol1)
    smiles_df["NumHBD"] =Chem.rdMolDescriptors.CalcNumHBD(mol1)
    smiles_df["NumHeteroatoms"] =Chem.rdMolDescriptors.CalcNumHeteroatoms(mol1)
    smiles_df["NumHeterocycles"] =Chem.rdMolDescriptors.CalcNumHeterocycles(mol1)
    smiles_df["NumLipinskiHBA"] =Chem.rdMolDescriptors.CalcNumLipinskiHBA(mol1)
    smiles_df["NumLipinskiHBD"] =Chem.rdMolDescriptors.CalcNumLipinskiHBD(mol1)
    smiles_df["NumRings"] =Chem.rdMolDescriptors.CalcNumRings(mol1)
    smiles_df["NumRotatableBonds"] =Chem.rdMolDescriptors.CalcNumRotatableBonds(mol1)
    smiles_df["NumSaturatedCarbocycles"] =Chem.rdMolDescriptors.CalcNumSaturatedCarbocycles(mol1)
    smiles_df["NumSaturatedHeterocycles"] =Chem.rdMolDescriptors.CalcNumSaturatedHeterocycles(mol1)
    smiles_df["NumSaturatedRings"] =Chem.rdMolDescriptors.CalcNumSaturatedRings(mol1)
    smiles_df["NumSpiroAtoms"] =Chem.rdMolDescriptors.CalcNumSpiroAtoms(mol1)
    smiles_df["PBF"] =Chem.rdMolDescriptors.CalcPBF(mol1)
    smiles_df["PMI1"] =Chem.rdMolDescriptors.CalcPMI1(mol1)
    smiles_df["PMI2"] =Chem.rdMolDescriptors.CalcPMI2(mol1)
    smiles_df["PMI3"] =Chem.rdMolDescriptors.CalcPMI3(mol1)
    smiles_df["TPSA"] =Chem.rdMolDescriptors.CalcTPSA(mol1)

    #surface properties from  
    ##PEOE =  The Partial Equalization of Orbital Electronegativities method of calculating atomic partial charges
    smiles_df["PEOE_VAS1"]=Chem.MolSurf.PEOE_VSA1(mol1)
    smiles_df["PEOE_VAS2"]=Chem.MolSurf.PEOE_VSA2(mol1)
    smiles_df["PEOE_VAS3"]=Chem.MolSurf.PEOE_VSA3(mol1)
    smiles_df["PEOE_VAS4"]=Chem.MolSurf.PEOE_VSA4(mol1)
    smiles_df["PEOE_VAS5"]=Chem.MolSurf.PEOE_VSA5(mol1)
    smiles_df["PEOE_VAS6"]=Chem.MolSurf.PEOE_VSA6(mol1)
    smiles_df["PEOE_VAS7"]=Chem.MolSurf.PEOE_VSA7(mol1)
    smiles_df["PEOE_VAS8"]=Chem.MolSurf.PEOE_VSA8(mol1)
    smiles_df["PEOE_VAS9"]=Chem.MolSurf.PEOE_VSA9(mol1)
    smiles_df["PEOE_VAS10"]=Chem.MolSurf.PEOE_VSA10(mol1)
    smiles_df["PEOE_VAS11"]=Chem.MolSurf.PEOE_VSA11(mol1)
    smiles_df["PEOE_VAS12"]=Chem.MolSurf.PEOE_VSA12(mol1)
    smiles_df["PEOE_VAS13"]=Chem.MolSurf.PEOE_VSA13(mol1)
    smiles_df["PEOE_VAS14"]=Chem.MolSurf.PEOE_VSA14(mol1)
    ##SMR = Molecular refractivity
    smiles_df["SMR_VSA1"]=Chem.MolSurf.SMR_VSA1(mol1)
    smiles_df["SMR_VSA2"]=Chem.MolSurf.SMR_VSA2(mol1)
    smiles_df["SMR_VSA3"]=Chem.MolSurf.SMR_VSA3(mol1)
    smiles_df["SMR_VSA4"]=Chem.MolSurf.SMR_VSA4(mol1)
    smiles_df["SMR_VSA5"]=Chem.MolSurf.SMR_VSA5(mol1)
    smiles_df["SMR_VSA6"]=Chem.MolSurf.SMR_VSA6(mol1)
    smiles_df["SMR_VSA7"]=Chem.MolSurf.SMR_VSA7(mol1)
    smiles_df["SMR_VSA8"]=Chem.MolSurf.SMR_VSA8(mol1)
    smiles_df["SMR_VSA9"]=Chem.MolSurf.SMR_VSA9(mol1)
    smiles_df["SMR_VSA10"]=Chem.MolSurf.SMR_VSA10(mol1)
    ##slogp = Log of the octanol/water partition coefficient
    smiles_df["SlogP_VSA1"]=Chem.MolSurf.SlogP_VSA1(mol1)
    smiles_df["SlogP_VSA2"]=Chem.MolSurf.SlogP_VSA2(mol1)
    smiles_df["SlogP_VSA3"]=Chem.MolSurf.SlogP_VSA3(mol1)
    smiles_df["SlogP_VSA4"]=Chem.MolSurf.SlogP_VSA4(mol1)
    smiles_df["SlogP_VSA5"]=Chem.MolSurf.SlogP_VSA5(mol1)
    smiles_df["SlogP_VSA6"]=Chem.MolSurf.SlogP_VSA6(mol1)
    smiles_df["SlogP_VSA7"]=Chem.MolSurf.SlogP_VSA7(mol1)
    smiles_df["SlogP_VSA8"]=Chem.MolSurf.SlogP_VSA8(mol1)
    smiles_df["SlogP_VSA9"]=Chem.MolSurf.SlogP_VSA9(mol1)
    smiles_df["SlogP_VSA10"]=Chem.MolSurf.SlogP_VSA10(mol1)
    smiles_df["SlogP_VSA11"]=Chem.MolSurf.SlogP_VSA11(mol1)
    ##others
    smiles_df["pyLabuteASA"]=Chem.MolSurf.pyLabuteASA(mol1)

    #3D descriptors from Chem.Descriptors3D module
    smiles_df["Asphericity"] = Chem.Descriptors3D.Asphericity(mol1)
    smiles_df["Eccentricity"] = Chem.Descriptors3D.Eccentricity(mol1)
    smiles_df["InertialShapeFactor"] = Chem.Descriptors3D.InertialShapeFactor(mol1)
    smiles_df["RadiusOfGyration"] = Chem.Descriptors3D.RadiusOfGyration(mol1)
    smiles_df["SpherocityIndex"] = Chem.Descriptors3D.SpherocityIndex(mol1)
    
    return(smiles_df)

# Here we can add more molecular descriptors
class MoleculeDataset(Dataset):

    """
    Dataset class for creating molecular graphs.

    Attributes:
    - smile_df (pandas.DataFrame): DataFrame containing SMILES data.
    - smile_column (str): Name of the column containing SMILES strings.
    - id_column (str): Name of the column containing molecule IDs.
    """

    def __init__(self, smile_df, smile_column, id_column):
        super(Dataset, self).__init__()

        # Gather the SMILES and the corresponding IDs
        self.smiles_data = smile_df[smile_column].tolist()
        self.id_data = smile_df[id_column].tolist()

    def __getitem__(self, index):
        # Get the molecule
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        mol = Chem.AddHs(mol)

        #########################
        # Get the molecule info #
        #########################
        type_idx = []
        chirality_idx = []
        atomic_number = []

        # Roberto: Might want to add more features later on. Such as atomic spin
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                print(self.id_data[index])

            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                    chem_id=self.id_data[index])
        
        return data

    def __len__(self):
        return len(self.smiles_data)
    
    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return self.__len__()
    

# Function to generate the molecular scaffolds
def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


# Function to separate structures based on scaffolds
def generate_scaffolds(smile_list):

    """
    Generate molecular MURCKO scaffolds from a list of SMILES strings.

    Parameters:
    - smile_list (list): List of SMILES strings.

    Returns:
    - scaffold_sets (list): List of scaffold sets.
    """

    scaffolds = {}
    data_len = len(smile_list)

    print("About to generate scaffolds")
    for ind, smiles in enumerate(smile_list):
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets

# Separate train, validation and test sets based on scaffolds
def scaffold_split(data_df, valid_size, test_size, smile_column, id_column):
    """
    Split data based on molecular scaffolds.

    Parameters:
    - data_df (pandas.DataFrame): DataFrame containing data to split.
    - valid_size (float): Proportion of data to allocate for validation.
    - test_size (float): Proportion of data to allocate for testing.
    - smile_column (str): Name of the column containing SMILES strings.
    - id_column (str): Name of the column containing molecule IDs.

    Returns:
    - train_ids (list): List of molecule IDs for the training set.
    - valid_ids (list): List of molecule IDs for the validation set.
    - test_ids (list): List of molecule IDs for the test set.
    """

    # Determine molecular scaffolds
    dataset = data_df[smile_column].tolist()
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    # Determine splits
    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set

    # Gather chem_ids based on 
    chemical_ids = data_df[id_column].tolist()
    train_ids = [chemical_ids[ind] for ind in train_inds]
    valid_ids = [chemical_ids[ind] for ind in valid_inds]
    test_ids = [chemical_ids[ind] for ind in test_inds]

    return train_ids, valid_ids, test_ids
    

# Function to split the dataset
def split_dataset(smile_df, valid_size, test_size, split_strategy, smile_col, id_col):

    """
    Split dataset into training, validation, and test sets.

    Parameters:
    - smile_df (pandas.DataFrame): DataFrame containing SMILES data.
    - valid_size (float): Proportion of data to allocate for validation.
    - test_size (float): Proportion of data to allocate for testing.
    - split_strategy (str): Splitting strategy ("random" or "scaffold").
    - smile_col (str): Name of the column containing SMILES strings.
    - id_col (str): Name of the column containing molecule IDs.

    Returns:
    - splitted_smiles_df (pandas.DataFrame): DataFrame with split information.
    """

    # Determine the splitting strategy
    if split_strategy == "random":

        # Determine the number of samples
        n_samples = smile_df.shape[0]

        # Randomly shuffle the indices
        chem_ids = smile_df[id_col].values
        np.random.shuffle(chem_ids)

        # Grab the validation ids
        valid_split = int(np.floor(valid_size * n_samples))
        valid_ids = chem_ids[:valid_split]

        # Grab the test ids
        test_split = int(np.floor(test_size * n_samples))
        test_ids = chem_ids[valid_split:(valid_split + test_split)]

        # Grab the train ids
        train_ids = chem_ids[(valid_split + test_split):]

    elif split_strategy == "scaffold":
        train_ids, valid_ids, test_ids = scaffold_split(smile_df, valid_size, test_size, smile_column=smile_col, id_column=id_col)

    # Add column with split information
    smile_df["split"]  = smile_df[id_col].apply(lambda x: "train" if x in train_ids else "valid" if x in valid_ids else "test")

    return smile_df

# Function to generate the molecular representation with MolE
def batch_representation(smile_df, dl_model, column_str, id_str, batch_size= 10_000, id_is_str=True, device="cuda:0"):

    """
    Generate molecular representations using a Deep Learning model.

    Parameters:
    - smile_df (pandas.DataFrame): DataFrame containing SMILES data.
    - dl_model: Deep Learning model for molecular representation.
    - column_str (str): Name of the column containing SMILES strings.
    - id_str (str): Name of the column containing molecule IDs.
    - batch_size (int, optional): Batch size for processing (default is 10,000).
    - id_is_str (bool, optional): Whether IDs are strings (default is True).
    - device (str, optional): Device for computation (default is "cuda:0").

    Returns:
    - chem_representation (pandas.DataFrame): DataFrame containing molecular representations.
    """
    
    # First we create a list of graphs
    molecular_graph_dataset = MoleculeDataset(smile_df, column_str, id_str)
    graph_list = [g for g in molecular_graph_dataset]

    # Determine number of loops to do given the batch size
    n_batches = len(graph_list) // batch_size

    # Are all molecules accounted for?
    remaining_molecules = len(graph_list) % batch_size

    # Starting indices
    start, end = 0, batch_size

    # Determine number of iterations
    if remaining_molecules == 0:
        n_iter = n_batches
    
    elif remaining_molecules > 0:
        n_iter = n_batches + 1
    
    # A list to store the batch dataframes
    batch_dataframes = []

    # Iterate over the batches
    for i in range(n_iter):
        # Start batch object
        batch_obj = Batch()
        graph_batch = batch_obj.from_data_list(graph_list[start:end])
        graph_batch = graph_batch.to(device)

        # Gather the representation
        with torch.no_grad():
            dl_model.eval()
            h_representation, _ = dl_model(graph_batch)
            chem_ids = graph_batch.chem_id
        
        batch_df = pd.DataFrame(h_representation.cpu().numpy(), index=chem_ids)
        batch_dataframes.append(batch_df)

        # Get the next batch
        ## In the final iteration we want to get all the remaining molecules
        if i == n_iter - 2:
            start = end
            end = len(graph_list)
        else:
            start = end
            end = end + batch_size
    
    # Concatenate the dataframes
    chem_representation = pd.concat(batch_dataframes)

    return chem_representation

# Function to load a pre-trained model
def load_pretrained_model(pretrain_architecture, pretrained_model, pretrained_dir = "../pretrained_model", device="cuda:0"):

    """
    Load a pre-trained MolE model.

    Parameters:
    - pretrain_architecture (str): Architecture of the pre-trained model.
    - pretrained_model (str): Name of the pre-trained MolE model.
    - pretrained_dir (str, optional): Directory containing pre-trained models (default is "../pretrained_model").
    - device (str, optional): Device for computation (default is "cuda:0").

    Returns:
    - model: Loaded pre-trained model.
    """

    # Read model configuration
    config = yaml.load(open(os.path.join(pretrained_dir, pretrained_model, "config.yaml"), "r"), Loader=yaml.FullLoader)
    model_config = config["model"]

    # Instantiate model
    if pretrain_architecture == "gin_concat":
        from models.ginet_concat import GINet
        model = GINet(**model_config).to(device)
    
    # Load pre-trained weights
    model_pth_path = os.path.join(pretrained_dir, pretrained_model, "model.pth")
    print(model_pth_path)

    state_dict = torch.load(model_pth_path, map_location=device)
    model.load_my_state_dict(state_dict)

    return model

# Function to generate the ECFP4 as an array
def fp_array(fingerprin_object):

    """
    Convert fingerprint object to NumPy array.

    Parameters:
    - fingerprin_object: Fingerprint object.

    Returns:
    - array (numpy.ndarray): NumPy array representation of the fingerprint.
    """

    # Initialise an array full of zeros
    array = np.zeros((0,), dtype=np.int8)

    # Dump fingerprint info into array
    DataStructs.ConvertToNumpyArray(fingerprin_object, array)

    return array

# Function to generate the ECFP4 representation
def generate_fps(smile_df, smile_col, id_col):

    """
    Generate Extended-Connectivity Fingerprints (ECFP4) representations for molecules.

    Parameters:
    - smile_df (pandas.DataFrame): DataFrame containing SMILES data.
    - smile_col (str): Name of the column containing SMILES strings.
    - id_col (str): Name of the column containing molecule IDs.

    Returns:
    - fps_dataframe (pandas.DataFrame): DataFrame containing ECFP4 representations.
    """

    # Generate fingerprints
    mol_objs = [Chem.MolFromSmiles(smile) for smile in smile_df[smile_col].tolist()]
    fp_objs = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mol_objs]

    # Place fingerprints in array
    fps_arrays = [fp_array(fp) for fp in fp_objs]

    # Create dataframe
    fps_matrix = np.stack(fps_arrays, axis=0 )
    fps_dataframe = pd.DataFrame(fps_matrix, index=smile_df[id_col].tolist())

    return fps_dataframe

def generate_descriptors(smile_df, smile_col, id_col):

    """
    Generate molecular descriptors for molecules.

    Parameters:
    - smile_df (pandas.DataFrame): DataFrame containing SMILES data.
    - smile_col (str): Name of the column containing SMILES strings.
    - id_col (str): Name of the column containing molecule IDs.

    Returns:
    - chemdesc_df (pandas.DataFrame): DataFrame containing molecular descriptors.
    """


    # Iterate over chemicals, making sure to catch exceptions
    descriptor_list = []
    for smile, id in zip(smile_df[smile_col].tolist(), smile_df[id_col].tolist()):
        try:
             desc_df = calc_descriptors(smile, id)
             descriptor_list.append(desc_df)

        except:
            print(f"Could not compute descriptors for {id}")
        
        
    # Concatenate and output
    chemdesc_df = pd.concat([d for d in descriptor_list if type(d) != str])
    chemdesc_df = chemdesc_df.rename(columns = {"chem_id": id_col}).set_index(id_col)

    return chemdesc_df

# Main function to process the dataset
def process_dataset(dataset_path,
                    smile_column_str = "rdkit_no_salt", 
                    id_column_str = "prestwick_ID",
                    pretrain_architecture=None, 
                    pretrained_model = None, 
                    split_approach="scaffold", 
                    validation_proportion=0.1, 
                    test_proportion=0.1, 
                    dataset_split=True,
                    device="cuda:0"):
    
    """
    Process the dataset to generate molecular representations.

    Parameters:
    - dataset_path (str): Path to the dataset file.
    - pretrain_architecture (str): Architecture of the pre-trained model or method ("ECFP4", "ChemDesc", or custom).
    - pretrained_model (str): Name of the pre-trained model. Can also be "MolCLR" or "ECFP4".
    - split_approach (str, optional): Splitting approach ("scaffold" or "random") (default is "scaffold").
    - validation_proportion (float, optional): Proportion of data to allocate for validation (default is 0.1).
    - test_proportion (float, optional): Proportion of data to allocate for testing (default is 0.1).
    - smile_column_str (str, optional): Name of the column containing SMILES strings (default is "rdkit_no_salt").
    - id_column_str (str, optional): Name of the column containing molecule IDs (default is "prestwick_ID").
    - dataset_split (bool, optional): Whether to split the dataset into train, validation, and test sets (default is True).
    - device (str): Device to use for computation (default is "cuda:0"). Can also be "cpu".

    Returns:
    - splitted_smiles_df (pandas.DataFrame): DataFrame with split information if split_data=True.
    - udl_representation (pandas.DataFrame): DataFrame containing molecular representations if split_data=False.
    """

    # First we read in the smiles as a dataframe
    smiles_df = read_smiles(dataset_path, smile_col=smile_column_str, id_col=id_column_str)

    # The we split the dataset into train, validation and test
    if dataset_split:
        splitted_smiles_df = split_dataset(smiles_df, validation_proportion, test_proportion, split_approach, smile_column_str, id_column_str)

    # Determine the representation
    if pretrained_model == "ECFP4":
        udl_representation = generate_fps(smiles_df, smile_column_str, id_column_str)
        
    elif pretrained_model == "ChemDesc":
        udl_representation = generate_descriptors(smiles_df, smile_column_str, id_column_str)
    
    else:
        # Now we load our pretrained model
        pmodel = load_pretrained_model(pretrain_architecture, pretrained_model, device=device)
        # Obtain the requested representation
        udl_representation = batch_representation(smiles_df, pmodel, smile_column_str, id_column_str, device=device)

    if dataset_split:
        return splitted_smiles_df, udl_representation
    
    else:
        return udl_representation