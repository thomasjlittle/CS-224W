import copy
import torch
import deepsnap
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from sklearn.metrics import f1_score
from deepsnap.hetero_gnn import forward_op
from deepsnap.hetero_graph import HeteroGraph
from torch_sparse import SparseTensor, matmul

class HeteroGNN(torch.nn.Module):
    def __init__(self, hetero_graph, args, aggr="mean"):
    	super(HeteroGNN, self).__init__()

    	self.aggr = aggr
    	self.hidden_size = args['hidden_size']

    	self.bns1 = nn.ModuleDict()
    	self.bns2 = nn.ModuleDict()
    	self.relus1 = nn.ModuleDict()
    	self.relus2 = nn.ModuleDict()
    	self.post_mps = nn.ModuleDict()
    	
    	convs1 = generate_convs(hetero_graph=hetero_graph, conv=HeteroGNNConv, hidden_size=self.hidden_size, first_layer=True)
    	self.convs1 = HeteroGNNWrapperConv(convs=convs1, args=args, aggr=self.aggr)
    	convs2 = generate_convs(hetero_graph=hetero_graph, conv=HeteroGNNConv, hidden_size=self.hidden_size, first_layer=False)
    	self.convs2 = HeteroGNNWrapperConv(convs=convs2, args=args, aggr=self.aggr)
    	
    	for node_type in hetero_graph.node_types:
    	    self.bns1[node_type] = torch.nn.BatchNorm1d(self.hidden_size, eps=1)
    	    self.bns2[node_type] = torch.nn.BatchNorm1d(self.hidden_size, eps=1)
    	    self.relus1[node_type] = nn.LeakyReLU()
    	    self.relus2[node_type] = nn.LeakyReLU()
    	    
    	    self.post_mps[node_type] = nn.Linear(self.hidden_size, hetero_graph.num_node_labels(node_type))
    	    
    def forward(self, node_feature, edge_index):
    	x = node_feature
        x = self.convs1(x, edge_index)

        
        x = forward_op(x, self.bns1)
        x = forward_op(x, self.relus1)
        x = self.convs2(x, edge_index)
        x = forward_op(x, self.bns2)
        x = forward_op(x, self.relus2)
        x = forward_op(x, self.post_mps)
        
        return x

    def loss(self, preds, y, indices):
        loss = 0
        loss_func = F.cross_entropy

        for node_type in preds:
        	prediction = preds[node_type][indices[node_type]]
        	loss += loss_func(prediction, y[node_type][indices[node_type]])

        return loss

class HeteroGNNConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels_src, in_channels_dst, out_channels):
        super(HeteroGNNConv, self).__init__(aggr="mean")

        self.in_channels_src = in_channels_src
        self.in_channels_dst = in_channels_dst
        self.out_channels = out_channels

        self.lin_dst = nn.Linear(self.in_channels_dst, self.out_channels)
        self.lin_src = nn.Linear(self.in_channels_src, self.out_channels)
        self.lin_update = nn.Linear(self.out_channels * 2, self.out_channels)

    def forward(
        self,
        node_feature_src,
        node_feature_dst,
        edge_index,
        size=None
    ):

        return self.propagate(edge_index=edge_index, size = size, node_feature_src=node_feature_src, node_feature_dst=node_feature_dst)

    def message_and_aggregate(self, edge_index, node_feature_src):
        out = None
        out = matmul(edge_index, node_feature_src, reduce="mean")

        return out

    def update(self, aggr_out, node_feature_dst):
        aggr_out = self.lin_src(aggr_out)
        node_feature_dst = self.lin_dst(node_feature_dst)
        aggr_out = self.lin_update(torch.cat((node_feature_dst, aggr_out), dim=-1))

        return aggr_out


class HeteroGNNWrapperConv(deepsnap.hetero_gnn.HeteroConv):
    def __init__(self, convs, args, aggr="mean"):
        super(HeteroGNNWrapperConv, self).__init__(convs, None)
        self.aggr = aggr

        # Map the index and message type
        self.mapping = {}

        # A numpy array that stores the final attention probability
        self.alpha = None

        self.attn_proj = None

        if self.aggr == "attn":
            self.attn_proj = nn.Sequential(
                nn.Linear(args['hidden_size'], args['attn_size']),
                nn.Tanh(),
                nn.Linear(args['attn_size'], 1, bias=False)
            )
    
    def reset_parameters(self):
        super(HeteroConvWrapper, self).reset_parameters()
        if self.aggr == "attn":
            for layer in self.attn_proj.children():
                layer.reset_parameters()
    
    def forward(self, node_features, edge_indices):
        message_type_emb = {}
        for message_key, message_type in edge_indices.items():
            src_type, edge_type, dst_type = message_key
            node_feature_src = node_features[src_type]
            node_feature_dst = node_features[dst_type]
            edge_index = edge_indices[message_key]
            message_type_emb[message_key] = (
                self.convs[message_key](
                    node_feature_src,
                    node_feature_dst,
                    edge_index,
                )
            )
        node_emb = {dst: [] for _, _, dst in message_type_emb.keys()}
        mapping = {}        
        for (src, edge_type, dst), item in message_type_emb.items():
            mapping[len(node_emb[dst])] = (src, edge_type, dst)
            node_emb[dst].append(item)
        self.mapping = mapping
        for node_type, embs in node_emb.items():
            if len(embs) == 1:
                node_emb[node_type] = embs[0]
            else:
                node_emb[node_type] = self.aggregate(embs)
        return node_emb
    
    def aggregate(self, xs):

        if self.aggr == "mean":
            h = torch.stack(xs)
            return torch.mean(h, dim=0)

        elif self.aggr == "attn":
            N = xs[0].shape[0] # Number of nodes for that node type
            M = len(xs) # Number of message types for that node type

            x = torch.cat(xs, dim=0).view(M, N, -1) # M * N * D
            z = self.attn_proj(x).view(M, N) # M * N * 1
            z = z.mean(1) # M * 1
            alpha = torch.softmax(z, dim=0) # M * 1

            # Store the attention result to self.alpha as np array
            self.alpha = alpha.view(-1).data.cpu().numpy()
  
            alpha = alpha.view(M, 1, 1)
            x = x * alpha
            return x.sum(dim=0)

def generate_convs(hetero_graph, conv, hidden_size, first_layer=False):
	convs = {}

	for message_type in hetero_graph.message_types:
		if first_layer:
			src_node = message_type[0]
			dst_node = message_type[2]
			in_channel_src = hetero_graph.num_node_features(src_node)
			in_channel_dst = hetero_graph.num_node_features(dst_node)
			convs[message_type] = conv(in_channels_src=in_channel_src, in_channels_dst=in_channel_dst, out_channels=hidden_size)
		else:
			convs[message_type] = conv(in_channels_src=hidden_size, in_channels_dst=hidden_size, out_channels=hidden_size)

	return convs