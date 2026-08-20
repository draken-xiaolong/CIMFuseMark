#!/usr/bin/env python3
"""Plot geometry-identical CIM identity sensitivity."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

VARIANTS = [
    ("attribute_removed", "Attributes\nremoved"),
    ("semantic_relabelled", "Semantics\nrelabeled"),
    ("hierarchy_flattened", "Hierarchy\nflattened"),
    ("serialization_reordered_control", "XML order\n(control)"),
]
METHODS = [
    ("jiang18", "Jiang18 geometry", "#777777"),
    ("cimfusemark_semantic_channels_masked", "CIMFuseMark masked", "#5B8DB8"),
    ("cimfusemark_full", "CIMFuseMark full", "#C83E4D"),
]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--input", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); report = json.loads(Path(args.input).read_text())
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "pdf.fonttype": 42})
    x = np.arange(len(VARIANTS)); width = .23
    fig, ax = plt.subplots(figsize=(6.8, 3.15))
    for index, (method, label, color) in enumerate(METHODS):
        values = [report["methods"][method][variant]["mean_identity_separation"] for variant, _ in VARIANTS]
        bars = ax.bar(x + (index-1)*width, values, width, label=label, color=color, edgecolor="white", linewidth=.5)
        for bar, value in zip(bars, values):
            if value > .004: ax.text(bar.get_x()+bar.get_width()/2, value+.006, f"{value:.3f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels([label for _, label in VARIANTS]); ax.set_ylabel("Identity separation (1 − NC)")
    ax.set_ylim(0, .30); ax.grid(axis="y", color="#E5E5E5", linewidth=.6); ax.set_axisbelow(True)
    ax.spines[["right", "top"]].set_visible(False); ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_title("Geometry-identical CIM identity test (64 CityGML tiles)", fontweight="bold")
    fig.text(.5, .015, "Every coordinate is exactly unchanged; XML reordering is an invariance control.", ha="center", fontsize=7, color="#555")
    fig.tight_layout(rect=(0, .06, 1, 1)); stem = Path(args.output); stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": 400}), (".svg", {})):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", **kwargs)


if __name__ == "__main__": main()
