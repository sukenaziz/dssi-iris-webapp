"""
This module contains data loading and preprocessing procedures.
"""

import argparse
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

SPECIES_MAP = {0: "setosa", 1: "versicolor", 2: "virginica"}


def load_data(data_path):
    """Load raw CSV dataset."""
    df = pd.read_csv(data_path)
    return df


def preprocess(df):
    """
    Preprocess raw dataframe for inference.
    For Iris, features are already numeric — no transformation needed.
    """
    return df


def run(data_path):
    """
    Main entry point: load and preprocess data for training.
        Parameters:
            data_path (str): Path to iris.csv
        Returns:
            df: Preprocessed dataframe
    """
    logging.info("Loading data...")
    df = load_data(data_path)
    logging.info(f"Dataset shape: {df.shape}")
    df = preprocess(df)
    return df


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str)
    args = argparser.parse_args()
    run(args.data_path)
