from datasets import load_dataset, Dataset
from collections import defaultdict
import argparse
import random
import torch
import sys
import os
import torch
import torch_geometric
import networkx as nx
from deepsnap.hetero_graph import HeteroGraph
import matplotlib.pyplot as plt

sys.path.append(os.getcwd() + "/src/BCDB")
from SMILES import SMILES


PHASES = {
    "cylinder": 0,
    "disordered": 1,
    "gyroid": 2,
    "HPL": 3,
    "lamellar": 4,
    "PL": 5,
    "sphere": 6,
}

ATOMS = {
    "C": 0,
    "N": 1,
    "O": 2,
    "S": 3,
    "Si": 4,
    "c": 5,
    "n": 6,
    "o": 7,
    "s": 8,
    "si": 9,
    "Unknown": 10,
}

BONDS = {
    "": 0,
    "=": 1,
    "\\": 2,
    "/": 3,
}

# PHASES_ONEHOT = torch.nn.functional.one_hot(torch.arange(0, len(PHASES)), num_classes=-1)


def load_data(args):
    raw_dataset = load_dataset(path="csv", data_files=args.dataset_path, split="train")

    # Filter unnecesary columns out of dataset
    cols_included = set(["phase1", "phase2", "T", "BigSMILES", "Mn", "f1"])
    cols_excluded = set(raw_dataset.column_names) - cols_included
    dataset = raw_dataset.remove_columns(cols_excluded)

    # Filter out diblock polymers with uncommon phases
    dataset_filtered = dataset.filter(lambda x: x["phase1"] in PHASES.keys())

    return dataset_filtered


def split_data(dataset):
    # Group examples by their X_values
    x_values_groups = defaultdict(list)
    for example in dataset:
        x_values_groups[example["BigSMILES"]].append(example)

    # Split unique X_values into train, test, and dev sets
    unique_x_values = list(x_values_groups.keys())
    random.shuffle(unique_x_values)

    split_ratios = (0.7, 0.2, 0.1)
    num_train = int(len(unique_x_values) * split_ratios[0])
    num_test = int(len(unique_x_values) * split_ratios[1])

    train_x_values = unique_x_values[:num_train]
    test_x_values = unique_x_values[num_train : num_train + num_test]
    dev_x_values = unique_x_values[num_train + num_test :]

    # Combine the groups to create train, test, and dev datasets
    train_examples = [example for x in train_x_values for example in x_values_groups[x]]
    test_examples = [example for x in test_x_values for example in x_values_groups[x]]
    dev_examples = [example for x in dev_x_values for example in x_values_groups[x]]

    train_dataset = Dataset.from_list(train_examples)
    test_dataset = Dataset.from_list(test_examples)
    dev_dataset = Dataset.from_list(dev_examples)

    return train_dataset, dev_dataset, test_dataset


def BigSMILES_to_SMILES(input_strs):
    chars_to_remove = "{}[]$<>,"
    translation_table = str.maketrans("", "", chars_to_remove)
    input_strs = [s.translate(translation_table) for s in input_strs]
    input_strs = [s.replace("Si", "[Si]").replace("H", "[H]") for s in input_strs]
    return input_strs


def label_fxn_groups(bigsmiles, graphs):
    # Since functional groups are always at polymer ends in SMILES, find number of graph nodes
    # before and after polymer blocks and label all nodes in those ranges as functional groups
    for bigsmile, graph in zip(bigsmiles, graphs):
        node_idx = 1
        if bigsmile[0] != "{":
            start = bigsmile.find("{")
            for idx in range(start):
                if bigsmile[idx].isalpha():
                    graph._node[node_idx]["in_functional_group"] = 1
                    node_idx += 1

        node_idx = len(graph._node)
        if bigsmile[-1] != "}":
            end = bigsmile.rfind("}")
            for idx in range(len(bigsmile) - 1, end, -1):
                if bigsmile[idx].isalpha():
                    graph._node[node_idx]["in_functional_group"] = 1
                    node_idx -= 1

    return graphs


################################################### Start Maybe Unused #####################################################

# import packages
# general tools
import numpy as np

# RDkit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

# Pytorch and Pytorch Geometric
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader


def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [
        int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))
    ]
    return binary_encoding


def get_atom_features(atom, use_chirality=False, hydrogens_implicit=False):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """
    # define list of permitted atoms

    permitted_list_of_atoms = [
        "C",
        "N",
        "O",
        "S",
        "F",
        "Si",
        "Cl",
        "Br",
        "Unknown",
    ]

    if hydrogens_implicit == False:
        permitted_list_of_atoms = ["H"] + permitted_list_of_atoms

    # compute atom features

    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)

    n_heavy_neighbors_enc = one_hot_encoding(
        int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"]
    )

    formal_charge_enc = one_hot_encoding(
        int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"]
    )

    hybridisation_type_enc = one_hot_encoding(
        str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]
    )

    is_in_a_ring_enc = [int(atom.IsInRing())]

    is_aromatic_enc = [int(atom.GetIsAromatic())]

    atomic_mass_scaled = [float((atom.GetMass() - 10.812) / 116.092)]

    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6)]

    covalent_radius_scaled = [
        float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76)
    ]
    atom_feature_vector = (
        atom_type_enc
        + n_heavy_neighbors_enc
        + formal_charge_enc
        + hybridisation_type_enc
        + is_in_a_ring_enc
        + is_aromatic_enc
        + atomic_mass_scaled
        + vdw_radius_scaled
        + covalent_radius_scaled
    )

    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(
            str(atom.GetChiralTag()),
            ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"],
        )
        atom_feature_vector += chirality_type_enc

    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(
            int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"]
        )
        atom_feature_vector += n_hydrogens_enc

    # Add in_functional_group
    atom_feature_vector += [0]

    return np.array(atom_feature_vector)


def get_bond_features(bond, use_stereochemistry=True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """
    permitted_list_of_bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)

    bond_is_conj_enc = [int(bond.GetIsConjugated())]

    bond_is_in_ring_enc = [int(bond.IsInRing())]

    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(
            str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"]
        )
        bond_feature_vector += stereo_type_enc
    return np.array(bond_feature_vector)


def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y):
    """
    Inputs:

    x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
    y = [y_1, y_2, ...] ... a list of numerial labels for the SMILES strings (such as associated pKi values)

    Outputs:

    data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning

    """

    data_list = []

    for smiles, y_val in zip(x_smiles, y):
        # convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(smiles)
        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2 * mol.GetNumBonds()
        n_node_features = len(get_atom_features(mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(mol.GetBondBetweenAtoms(0, 1)))
        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)

        X = torch.tensor(X, dtype=torch.float)

        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim=0)

        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))

        for k, (i, j) in enumerate(zip(rows, cols)):
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))

        EF = torch.tensor(EF, dtype=torch.float)

        # construct label tensor
        y_tensor = torch.tensor(np.array([y_val]), dtype=torch.float)

        # construct Pytorch Geometric data object and append to data list
        data_list.append(Data(x=X, edge_index=E, edge_attr=EF, y=y_tensor))
    return data_list


################################################### End Maybe Unused #####################################################


# Convert to DeepSNAP Heterogeneous Graph


def set_graph_attrs(G, data, idx):
    """
    node_type = (str) atom name
    node_label = (int) representation of atom
    node_feature = (torch.tensor) list of [in_function_group, volume_fraction]
    edge_type = (str) name of bond
    edge_feature = (torch.tensor) list of [bond type (represented as int)]
    graph_feature =
    graph_label =
    """

    # Node Level Features
    # Generate Node Level Features
    node_types = {node: G._node[node]["atom"] for node in G.nodes()}
    node_labels = {node: ATOMS[node_types[node]] for node in G.nodes()}
    node_features = {
        node: torch.tensor([G._node[node]["in_functional_group"], data["f1"][idx]])
        for node in G.nodes()
    }

    # Clear unnecesary node atributes
    for n, d in G.nodes(data=True):
        d.clear()

    # Set Node Level Features
    nx.set_node_attributes(G, node_types, "node_type")
    nx.set_node_attributes(G, node_labels, "node_label")
    nx.set_node_attributes(G, node_features, "node_feature")

    # Node Level Features
    # First get dictionary of edge types:
    edge_dict = {}
    for i, inner_dict in G.edges()._adjdict.items():
        for j, properties in inner_dict.items():
            edge_dict[(i, j)] = properties["type"]
        edge_types = {}

    # Clear unnecessary edge attributes
    for n1, n2, d in G.edges(data=True):
        d.clear()

    # Now set edge types
    for edge in G.edges():
        edge_types[edge] = BONDS[edge_dict[edge]]
    nx.set_edge_attributes(G, edge_types, "edge_type")

    # Set Graph Level Features
    graph_feature = torch.tensor(data["Mn"][idx])

    # Visualize Graph
    # labels = nx.get_node_attributes(G, "node_type")
    # nx.draw(G, cmap=plt.get_cmap("coolwarm"), labels=labels)

    G_hete = HeteroGraph(G)

    return G_hete


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/diblock.csv")
    args = parser.parse_args()

    # Load and split data
    data = load_data(args)
    train, dev, test = split_data(data)

    # Extract BigSMILES and SMILES strings
    BigSmiles = train["BigSMILES"]
    smiles = BigSMILES_to_SMILES(BigSmiles)

    # Remove unsupported polymers
    smiles = [smile for smile in smiles if all(x not in smile for x in ["sn", "Sn"])]

    # Convert smiles into graphs and labels
    smiles_obj = [SMILES(smile) for smile in smiles]
    smiles_graphs = [obj.parse() for obj in smiles_obj]
    labels = [PHASES[label] for label in data["phase1"]]
    # graphs = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(smiles, labels)

    # Label functional groups
    graphs = label_fxn_groups(BigSmiles, smiles_graphs)
    G_hete = []
    for idx, graph in enumerate(graphs):
        G_hete.append = set_graph_attrs(graph, train, idx)
    print("To be completed ..")
