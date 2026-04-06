import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help=".pt or .csv")
    parser.add_argument("--point_size", type=float, default=2.0)
    parser.add_argument("--plane", type=str, default="xy", choices=["xy", "xz", "yz"])
    parser.add_argument("--save", type=str, default="", help="save figure path")
    args = parser.parse_args()

    ext = os.path.splitext(args.input)[1].lower()

    if ext == ".pt":
        x = torch.load(args.input)
        xyz = x["xyz"].cpu().numpy()
        scores = x["scores"].cpu().numpy()
        title = f'Visibility heatmap: {x.get("image_name", "N/A")}'

    elif ext == ".csv":
        df = pd.read_csv(args.input)
        xyz = df[["x", "y", "z"]].to_numpy()
        scores = df["score"].to_numpy()
        title = f"Visibility heatmap: {os.path.basename(args.input)}"

    else:
        raise ValueError("input must be .pt or .csv")

    if args.plane == "xy":
        a, b = 0, 1
        xlabel, ylabel = "x", "y"
    elif args.plane == "xz":
        a, b = 0, 2
        xlabel, ylabel = "x", "z"
    else:
        a, b = 1, 2
        xlabel, ylabel = "y", "z"

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(xyz[:, a], xyz[:, b], c=scores, s=args.point_size)
    plt.colorbar(sc, label="visibility score")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=200)
        print(f"saved: {args.save}")
    else:
        plt.show()

if __name__ == "__main__":
    main()