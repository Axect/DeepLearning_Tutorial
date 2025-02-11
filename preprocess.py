import polars as pl
import numpy as np


def split_data(df, ratio=0.8):
    """
    Split the data into training and validating sets.
    """
    length = len(df)
    ics = np.random.choice(length, int(length * ratio), replace=False)
    mask = np.zeros(length, dtype=bool)
    mask[ics] = True
    return df.filter(mask), df.filter(~mask)


if __name__ == "__main__":
    # Load the data
    df = pl.read_parquet("data/data.parquet")

    # Split the data
    train, val = split_data(df)

    # Save the data
    train.write_parquet("data/train.parquet")
    val.write_parquet("data/val.parquet")

    print(train)
    print(val)
    print("Data split and saved.")
