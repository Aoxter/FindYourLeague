import pandas as pd


def get_leagues_data(filename):
    df = pd.read_csv(filename, sep=";", header=0)
    return [row for _, row in df.iterrows()]

