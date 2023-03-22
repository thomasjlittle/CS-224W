import os
import torch
from torch_geometric.data import InMemoryDataset, download_url
from deepsnap.hetero_graph import HeteroGraph


class BCDBDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(BCDBDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["train_graphs.pt", "val_graphs.pt", "test_graphs.pt"]
