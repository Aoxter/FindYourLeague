import pandas as pd


def get_leagues_data():
    df = pd.read_csv('leagues.csv', sep=";", header=0)
    return [row for _, row in df.iterrows()]

