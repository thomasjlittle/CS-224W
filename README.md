# Predicting Copolymer Properties Using Graph Neural Networks

The idea from our experiment is to be able to recreate diblock microphrase separation plot. We wanted to be able to use Graph Neural Network Architecture to predict Block Copolymer phases. The best course of action to find a potential best model for our task using GraphGym to run multiple experiments using pre-built models. The BCDB data was turned into deepsnap graphs, where each graph represented a Polymer in a different phase. 

- GraphGym folder was forked from GraphGym repo
- BCDB database class was added in GraphGym/BCDB_Database/
- The BCDB graphs from the preprocessors.py script was saved to .pkl in run/datasets (I let GraphGym handle the split so that file is the full dataset)

## Data processing

In order to be able to turn the polymers into deepsnap graphs, we needed to process the data from our strings of Polymer chains into SMILE objects. We simplified the BigSMILE objects into SMILE objects due to our time constraints. Our polymers were processed so that our node types were strings of the atoms and that our edge types were bonds between the molecules. 

## Database

[Block Copolymer Phase Behavior Database(BCDB)](https://github.com/olsenlabmit/BCDB) contains 5300 entries of 61 different polymers at different temperatures and molar mass at different phases. 

## Developing locally

Since we forked from the Stanford Network Analysis Platform(snap) [repository](https://github.com/snap-stanford/GraphGym/blob/master/README.md), we run our code similar to them. We use a custom config file `BCDB_Dataset.yaml` stored under `run/configs` which specified our. We created a custom grid file under `run/grids` called `BCDB_grid.txt`. 

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

