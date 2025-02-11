import torch
import torch.functional as F
from torch.utils.data import TensorDataset, DataLoader
import polars as pl
import numpy as np
import argparse
import matplotlib.pyplot as plt
import scienceplots

from model import MLP


def load_data():
    df_train = pl.read_parquet("data/train.parquet")
    df_val = pl.read_parquet("data/val.parquet")
    return df_train, df_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    df_train, df_val = load_data()
    x_train = torch.tensor(df_train["x"].to_numpy(), dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(df_train["y"].to_numpy(), dtype=torch.float32).unsqueeze(1)
    x_val = torch.tensor(df_val["x"].to_numpy(), dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(df_val["y"].to_numpy(), dtype=torch.float32).unsqueeze(1)

    ds_train = TensorDataset(x_train, y_train)
    ds_val = TensorDataset(x_val, y_val)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size)

    hparams = {
        "nodes": args.nodes,
        "layers": args.layers
    }

    model = MLP(hparams)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    model.to(args.device)

    for epoch in range(args.epochs):
        model.train()
        for x, y in dl_train:
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            val_loss = 0
            for x, y in dl_val:
                x, y = x.to(args.device), y.to(args.device)
                y_hat = model(x)
                val_loss += criterion(y_hat, y)
            val_loss /= len(dl_val)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Val Loss: {val_loss.item()}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved!")

    model.eval()
    df_true = pl.read_parquet("./data/true.parquet")
    x_true = df_true["x"].to_numpy()
    y_true = df_true["y"].to_numpy()

    df_data = pl.read_parquet("./data/data.parquet")
    x_data = df_data["x"].to_numpy()
    y_data = df_data["y"].to_numpy()

    x_test = torch.linspace(0, 1, 3000).unsqueeze(1)
    y_test = model(x_test.to(args.device)).cpu().detach().numpy().squeeze()
    x_test = x_test.cpu().numpy().squeeze()

    rmse = np.sqrt(np.mean((y_true - y_test) ** 2))

    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots()
        ax.plot(x_data, y_data, '.', label="Data", color='blue', markersize=2, markeredgewidth=0, alpha=0.5)
        ax.plot(x_true, y_true, label="True", color='red')
        ax.plot(x_test, y_test, '--', label="Predicted", color='orange')
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"RMSE: {rmse:.4e}")
        ax.autoscale(tight=True)
        fig.savefig("plot.png", dpi=600, bbox_inches="tight")
