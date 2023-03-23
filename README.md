# Predicting Copolymer Properties Using Graph Neural Networks

The idea from our experiment is to be able to recreate diblock microphrase separation plot. We wanted to be able to use Graph Neural Network Architecture to predict Block Copolymer phases. The best course of action to find a potential best model for our task using GraphGym to run multiple experiments using pre-built models. The BCDB data was turned into deepsnap graphs, where each graph represented a Polymer in a different phase. We had the phases of each of these graphs, which we used as the labels for our prediction. 

The `GraphGym` folder is forked from this [repository](https://github.com/snap-stanford/GraphGym/blob/master/README.md), and we use GraphGym to test GCN, GAT, and GIN to see which model would perform best for our data. 

## Data processing

In order to be able to turn the polymers into deepsnap graphs, we needed to process the data from our strings of Polymer chains into SMILE objects. We simplified the BigSMILE objects into SMILE objects due to our time constraints. Our polymers were processed so that our node types were strings of the atoms and that our edge types were bonds between the molecules. We included the temperature of the polymer as well as the molar mass into our graph object as well. 

After processing the data, we placed created a database using the deepsnap graphs into `BCDB.pkl` file. This allowed us to be able to use the databse with GraphGym. GraphGym did not support custom splits at the time, so we simply allowed it to splot our graphs into training, dev, and test sets with a 80% train, 10% dev, and 10% test split. For our labels we used the polymer's phase classification for each of the graphs.

All of this processing was done by running the following script in the root directory after having activated the conda environment specified in the `environment.yml` file in the root directory.

```bash
python ./src/data_preprocessors.py
```

## Database

[Block Copolymer Phase Behavior Database(BCDB)](https://github.com/olsenlabmit/BCDB) contains 5300 entries of 61 different polymers at different temperatures and molar mass at different phases. As mentioned above, after we processed our data we saved it but we also added a BCDB database class under `GraphGym/BCDB_Databse/BCDB_dataset.py`.

### Using it in GraphGym

In order to use our custom dataset we specified the path of our dataset in our config file

## Developing locally

Since we forked from the Stanford Network Analysis Platform(snap) [repository](https://github.com/snap-stanford/GraphGym/blob/master/README.md), we run our code similar to them. We use a custom config file `BCDB_Dataset.yaml` stored under `run/configs` which specified our parameters for our GNN models. In order to experiment with the GNN settings and models we would edit this config file with our desired parameters. The bash script that we edited from the original GraphGym directory uses the config file to create the environment and specify the settings for the models that it tests. We also created a custom grid file under `run/grids` called `BCDB_grid.txt`. 

```bash
# The recommended basic settings for GNN
out_dir: results
device: cpu
dataset:
  format: nx
  dir: ./datasets
  name: BCDB
  task: graph
  task_type: classification
  transductive: False
  split: [0.8, 0.1, 0.1]
train:
  batch_size: 32
  eval_period: 20
  ckpt_period: 100
model:
  type: gnn
  loss_fun: cross_entropy
  edge_decoding: dot
  graph_pooling: add
gnn:
  layers_pre_mp: 1
  layers_mp: 2
  layers_post_mp: 1
  dim_inner: 256
  layer_type: gcnconv
  stage_type: stack
  batchnorm: True
  act: prelu
  dropout: 0.0
  agg: add
  normalize_adj: False
optim:
  optimizer: adam
  base_lr: 0.01
  max_epoch: 100
```

### Running the code: 

For grid batch we run:

```bash
cd GraphGym/run
bash run_batch.sh 
```

or if you would like to run a single batch then: 

```bash
cd GraphGym/run
sh run_single.sh (to run single)
```

