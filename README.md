# Predicting Copolymer Properties Using Graph Neural Networks - UNFINISHED

General idea - recreate diblock microphase separation plot as baseline and then expand to triblock polymers if possible


- GraphGym folder was forked from GraphGym repo
- BCDB database class was added in GraphGym/BCDB_Database/
- The BCDB graphs from the preprocessors.py script was saved to .pkl in run/datasets (I let GraphGym handle the split so that file is the full dataset)

## Database

## Data processing

## Developing locally

Since we forked from the Stanford Network Analysis Platform(snap) [repository](https://github.com/snap-stanford/GraphGym/blob/master/README.md), we run our code similar to them. We use a custom config file `BCDB_Dataset.yaml` stored under `run/configs` which specified our... We created a custom grid file under `run/grids` called `BCDB_grid.txt`... 

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

((Prob double check that these work since we put GraphGym folder) inside the main folder))


https://github.com/snap-stanford/GraphGym -- Tutorial/ how to run defaults
