import pandas as pd
import torch
from heteroGNN import HeteroGNN
from deepsnap.dataset import GraphDataset
from torch_geometric.loader import DataLoader
from torch_sparse import SparseTensor, matmul

def train(model, optimizer, hetero_graph, train_idx):
    model.train()
    optimizer.zero_grad()
    preds = model(hetero_graph.node_feature, hetero_graph.edge_index)

    loss = model.loss(preds=preds, y=hetero_graph.node_label, indices=train_idx)

    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, graph, indices, best_model=None, best_val=0, save_preds=False, agg_type=None):
    model.eval()
    accs = []
    for i, index in enumerate(indices):
        preds = model(graph.node_feature, graph.edge_index)
        num_node_types = 0
        micro = 0
        macro = 0
        for node_type in preds:
            idx = index[node_type]
            pred = preds[node_type][idx]
            pred = pred.max(1)[1]
            label_np = graph.node_label[node_type][idx].cpu().numpy()
            pred_np = pred.cpu().numpy()
            micro = f1_score(label_np, pred_np, average='micro')
            macro = f1_score(label_np, pred_np, average='macro')
            num_node_types += 1
                  
        micro /= num_node_types
        macro /= num_node_types
        accs.append((micro, macro))

        # Only save the test set predictions and labels!
        if save_preds and i == 2:
            print ("Saving Heterogeneous Node Prediction Model Predictions with Agg:", agg_type)
            print()

            data = {}
            data['pred'] = pred_np
            data['label'] = label_np

            df = pd.DataFrame(data=data)
            df.to_csv('ACM-Node-' + agg_type + 'Agg.csv', sep=',', index=False)

    if accs[1][0] > best_val:
        best_val = accs[1][0]
        best_model = copy.deepcopy(model)
    return accs, best_model, best_val

def run_train_test(train_graphs, dev_graphs, test_graphs, train_labels, dev_labels, test_labels):
    args = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'hidden_size': 64,
        'epochs': 100,
        'weight_decay': 1e-5,
        'lr': 0.003,
        'attn_size': 32,
    }

    print("Device: {}".format(args['device']))

    train_graph_dataset = GraphDataset(train_graphs)
    dev_graph_dataset = GraphDataset(dev_graphs)
    test_graph_dataset = GraphDataset(test_graphs)

    train_loader = DataLoader(train_graph_dataset)

    for hetero_graph in train_graphs:


    # # Load the train, validation and test indices
        train_idx = None # {"paper": data['train_idx'].to(args['device'])}
        val_idx = None # {"paper": data['val_idx'].to(args['device'])}
        test_idx = None # {"paper": data['test_idx'].to(args['device'])}


        # Node feature and node label to device
        for key in hetero_graph.node_feature:
            hetero_graph.node_feature[key] = hetero_graph.node_feature[key].to(args['device'])
        for key in hetero_graph.node_label:
            hetero_graph.node_label[key] = hetero_graph.node_label[key].to(args['device'])

        total_nodes = 0

        node_dict = hetero_graph.num_nodes()
        for node in node_dict:
            total_nodes += node_dict[node]

        # Edge_index to sparse tensor and to device
        for key in hetero_graph.edge_index:
            edge_index = hetero_graph.edge_index[key]
            adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(total_nodes, total_nodes))
            hetero_graph.edge_index[key] = adj.t().to(args['device'])



    # Mean Aggregation 
    best_model = None
    best_val = 0

    model = HeteroGNN(hetero_graph, args, aggr="mean").to(args['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    for epoch in range(args['epochs']):
        loss = train(model, optimizer, hetero_graph, train_idx)
        accs, best_model, best_val = test(model, hetero_graph, [train_idx, val_idx, test_idx], best_model, best_val)
        print(
          f"Epoch {epoch + 1}: loss {round(loss, 5)}, "
          f"train micro {round(accs[0][0] * 100, 2)}%, train macro {round(accs[0][1] * 100, 2)}%, "
          f"valid micro {round(accs[1][0] * 100, 2)}%, valid macro {round(accs[1][1] * 100, 2)}%, "
          f"test micro {round(accs[2][0] * 100, 2)}%, test macro {round(accs[2][1] * 100, 2)}%"
        )
    best_accs, _, _ = test(best_model, hetero_graph, [train_idx, val_idx, test_idx], save_preds=True, agg_type="Mean")
    print(
      f"Best model: "
      f"train micro {round(best_accs[0][0] * 100, 2)}%, train macro {round(best_accs[0][1] * 100, 2)}%, "
      f"valid micro {round(best_accs[1][0] * 100, 2)}%, valid macro {round(best_accs[1][1] * 100, 2)}%, "
      f"test micro {round(best_accs[2][0] * 100, 2)}%, test macro {round(best_accs[2][1] * 100, 2)}%"
    )

    # Attention Aggregation 
    best_model = None
    best_val = 0

    output_size = hetero_graph.num_node_labels('paper')
    model = HeteroGNN(hetero_graph, args, aggr="attn").to(args['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    for epoch in range(args['epochs']):
        loss = train(model, optimizer, hetero_graph, train_idx)
        accs, best_model, best_val = test(model, hetero_graph, [train_idx, val_idx, test_idx], best_model, best_val)
        print(
          f"Epoch {epoch + 1}: loss {round(loss, 5)}, "
          f"train micro {round(accs[0][0] * 100, 2)}%, train macro {round(accs[0][1] * 100, 2)}%, "
          f"valid micro {round(accs[1][0] * 100, 2)}%, valid macro {round(accs[1][1] * 100, 2)}%, "
          f"test micro {round(accs[2][0] * 100, 2)}%, test macro {round(accs[2][1] * 100, 2)}%"
        )
    best_accs, _, _ = test(best_model, hetero_graph, [train_idx, val_idx, test_idx], save_preds=True, agg_type="Attention")
    print(
      f"Best model: "
      f"train micro {round(best_accs[0][0] * 100, 2)}%, train macro {round(best_accs[0][1] * 100, 2)}%, "
      f"valid micro {round(best_accs[1][0] * 100, 2)}%, valid macro {round(best_accs[1][1] * 100, 2)}%, "
      f"test micro {round(best_accs[2][0] * 100, 2)}%, test macro {round(best_accs[2][1] * 100, 2)}%"
    )