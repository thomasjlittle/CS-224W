# Predicting Copolymer Properties Using Graph Neural Networks - UNFINISHED

General idea - recreate diblock microphase separation plot as baseline and then expand to triblock polymers if possible


- GraphGym folder was forked from GraphGym repo
- BCDB database class was added in GraphGym/BCDB_Database/
- The BCDB graphs from the preprocessors.py script was saved to .pkl in run/datasets (I let GraphGym handle the split so that file is the full dataset)
- A config file was created in run/configs
- A grid file was created in run/grids


To run the code: 
cd GraphGym/run
sh run_batch.sh (to run grid batch)

or 
sh run_single.sh (to run single)

((Prob double check that these work since we put GraphGym folder) inside the main folder))


https://github.com/snap-stanford/GraphGym -- Tutorial/ how to run defaults
