#!/usr/bin/env bash

# Test for running a single experiment. --repeat means run how many different random seeds.
python main.py --cfg configs/BCDB_Dataset.yaml --repeat 3
