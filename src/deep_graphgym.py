import graphgym
from deepsnap.hetero_graph import HeteroGraph
import torch
from torch_geometric.nn import GCNConv
from graphgym.models.gnn import GNN


def convert_to_pyg(graph_list):
	pyg_list = []
	names = []
	print(graph_list)
	for graph in graph_list:
		print(graph)
		pyg_list.append(graph.to_data_list())
		names.append(graph.graph_label)
	return pyg_list, names

def create_datasets_from_graphs(graph_list, names):
	datasets_list = []

	for idx, graph in enumerate(graph_list):
		dt = graphgym.create_dataset(names[idx], graph, task_type="graph_classification")
		datasets_list.append(dt)

	return datasets_list

def create_tasks(datasets_list):
	tasks_list = []
	for dataset in datasets_list:
		task = graphgym.GraphClassificationTask(GCN(), dataset)
		tasks_list.append(task)
	return tasks_list

def run_pipeline(tasks):
	pipeline = graphgym.Pipeline(tasks)
	pipeline.run()

def run_defined_experiments(graph_list):
	pyg_list, names = convert_to_pyg(graph_list)
	datasets = create_datasets_from_graphs(pyg_list, names)
	tasks = create_tasks(datasets)

	run_pipeline(tasks)