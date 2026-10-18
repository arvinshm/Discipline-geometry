"""Average the saved vectors and plot their full-dimensional Euclidean distances."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def mean_vectors(activations, labels, count):
    return np.stack([activations[labels == i].mean(axis=0, dtype=np.float64)
                     for i in range(count)])


def euclidean_distances(means):
    differences = means[:, None, :] - means[None, :, :]
    return np.sqrt(np.sum(differences ** 2, axis=-1))


def compare_distances(distances, reference, names):
    """Compare the current distances with the saved matrix from the previous run."""
    pairs = np.triu_indices(len(names), k=1)  # Each distinct pair appears once.
    old, new = reference[pairs], distances[pairs]
    correlation = np.corrcoef(old, new)[0, 1]
    median_change = np.median((new - old) / old)
    diagonal = np.eye(len(names), dtype=bool)
    old_order = np.argsort(np.where(diagonal, np.inf, reference), axis=1)
    new_order = np.argsort(np.where(diagonal, np.inf, distances), axis=1)
    same_nearest = old_order[:, 0] == new_order[:, 0]
    shared = sum(len(set(a[:3]) & set(b[:3])) for a, b in zip(old_order, new_order))
    print(f"Previous run comparison over {len(old)} pairs: Pearson r={correlation:.6f}")
    print(f"Median relative distance change: {median_change:+.2%}")
    print(f"Same nearest neighbor: {same_nearest.sum()}/{len(names)}; "
          f"shared top-three neighbors: {shared}/{3 * len(names)}")
    for i in np.flatnonzero(~same_nearest):
        print(f"  {names[i]}: {names[old_order[i, 0]]} -> {names[new_order[i, 0]]}")


def plot_distances(distances, names, path):
    fig, ax = plt.subplots(figsize=(13, 11), layout="constrained")
    im = ax.imshow(distances, cmap="viridis", vmin=0)
    ax.set_xticks(range(len(names)), names, rotation=55, ha="right", fontsize=9)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    ax.set_title(
        "Qwen3-4B-Base: Euclidean distances between discipline means",
        fontsize=13, pad=18,
    )
    colorbar = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.03)
    colorbar.set_label("Euclidean distance in raw activation units", fontsize=11)
    fig.supxlabel("Each cell uses all 2,560 activation coordinates; axes list disciplines.",
                  fontsize=10)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data.npz")
    parser.add_argument("--out", type=Path, default=ROOT,
                        help="Directory for distances.csv and distances.png")
    args = parser.parse_args()
    with np.load(args.data, allow_pickle=False) as saved:
        activations, labels = saved["activations"], saved["discipline"]
        names = saved["names"].tolist()
        reference = saved["reference_distances"] if "reference_distances" in saved else None
    counts = np.bincount(labels, minlength=len(names))
    assert activations.shape == (len(labels), 2560) and np.isfinite(activations).all()
    assert len(counts) == len(names) and counts.min() > 0 and counts.min() == counts.max()
    means = mean_vectors(activations, labels, len(names))
    distances = euclidean_distances(means)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "distances.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["discipline", *names])
        writer.writerows([name, *row] for name, row in zip(names, distances))
    plot_distances(distances, names, args.out / "distances.png")
    index = names.index("mathematics")
    nearest = [i for i in np.argsort(distances[index]) if i != index][:3]
    print("Mathematics nearest neighbors: " + ", ".join(
        f"{names[i]} ({distances[index, i]:.6f})" for i in nearest))
    if reference is not None:
        compare_distances(distances, reference, names)
    print(f"Saved the {len(names)} x {len(names)} distance matrix and heatmap to {args.out}")


if __name__ == "__main__":
    main()
