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

    # Labels:
    # 0 = in polymer B block
    # 1 = in functional group
    # 2 = in polymer A block

    for bigsmile, graph in zip(bigsmiles, graphs):
        node_idx = 1

        # Label nodes in left functional group
        start = bigsmile.find("{")
        if bigsmile[0] != "{":
            for idx in range(start):
                if bigsmile[idx].isalpha():
                    graph._node[node_idx]["in_functional_group"] = 1
                    node_idx += 1

        # Label nodes in polymer A block
        end_A_block = bigsmile.find("}")
        for idx in range(start + 1, end_A_block):
            if bigsmile[idx].isalpha():
                graph._node[node_idx]["in_functional_group"] = 2
                node_idx += 1

        # Label nodes in right functional group
        node_idx = len(graph._node)
        if bigsmile[-1] != "}":
            end = bigsmile.rfind("}")
            for idx in range(len(bigsmile) - 1, end, -1):
                if bigsmile[idx].isalpha():
                    graph._node[node_idx]["in_functional_group"] = 1
                    node_idx -= 1

    return graphs

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
        node: torch.tensor([G._node[node]["in_functional_group"], 1 - data["f1"][idx]])
        if G._node[node]["in_functional_group"] == 0
        else torch.tensor([G._node[node]["in_functional_group"], 0])
        if G._node[node]["in_functional_group"] == 1
        else torch.tensor([G._node[node]["in_functional_group"], data["f1"][idx]])
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

    # Convert smiles into graphs and labels
    smiles_obj = [SMILES(smile) for smile in smiles]
    smiles_graphs = [obj.parse() for obj in smiles_obj]
    labels = [PHASES[label] for label in data["phase1"]]
    # graphs = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(smiles, labels)

    # Label functional groups
    graphs = label_fxn_groups(BigSmiles, smiles_graphs)
    G_hete = []
    for idx, graph in enumerate(graphs):
        G_hete_graph = set_graph_attrs(graph, train, idx)
        G_hete.append(G_hete_graph)
    print("To be completed ..")
