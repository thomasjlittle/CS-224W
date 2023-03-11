from datasets import load_dataset
import argparse
from BCDB.BigSMILES_BigSmilesObj import BigSMILES


def load_data(args):
    raw_dataset = load_dataset(
        path="csv",
        data_files=args.dataset_path,
    )

    # Filter unnecesary columns out of dataset
    cols_included = set(["phase1", "phase2", "T", "BigSMILES", "Mn", "f1"])
    cols_excluded = set(raw_dataset.column_names["train"]) - cols_included
    dataset = raw_dataset.remove_columns(cols_excluded)

    # Filter out diblock polymers with uncommon phases
    phases_included = ["lamellar", "disordered", "cylinder", "HPL", "gyroid", "PL", "sphere"]
    dataset = dataset.filter(lambda x: x["phase1"] in phases_included)

    return dataset


# def convert_BigSMILES_to_graph(dataset):


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/diblock.csv")
    args = parser.parse_args()

    data = load_data(args)
