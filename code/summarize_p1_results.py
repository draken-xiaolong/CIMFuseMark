#!/usr/bin/env python3
"""Generate publication-ready P1 ablation, uncertainty and efficiency artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


LABELS = {
    "full_seed2026": "Full",
    "loss_no_contrastive": "w/o contrastive",
    "loss_no_bit_stability": "w/o bit stability",
    "loss_no_tail": "w/o tail loss",
    "loss_no_separation": "w/o separation",
    "loss_no_curriculum": "w/o curriculum",
    "loss_clean_only": "clean only",
    "feature_geometry": "geometry",
    "feature_geometry_type": "geometry + type",
    "feature_geometry_attributes": "geometry + attributes",
    "feature_geometry_depth": "geometry + depth",
    "feature_geometry_boundary": "geometry + boundary",
    "graph_untyped": "untyped",
    "graph_no_edges": "no edge / DeepSets",
    "graph_hierarchy_only": "hierarchy only",
    "graph_no_spatial": "w/o spatial",
    "graph_no_reverse": "w/o reverse",
    "graph_edge_drop_20": "edge drop 20%",
    "graph_edge_drop_40": "edge drop 40%",
    "graph_edge_drop_60": "edge drop 60%",
    "graph_random_rewire": "random rewire",
    "graph_hierarchy_flattened": "hierarchy flattened",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_ablation(rows: list[dict], group: str, output: Path) -> None:
    selected = [row for row in rows if row["group"] == group]
    reference = next(row for row in rows if row["id"] == "full_seed2026")
    selected = [reference, *selected]
    labels = [LABELS.get(row["id"], row["id"]) for row in selected]
    aucs = [float(row["auc_mean"]) for row in selected]
    tars = [float(row["tar_far5_mean"]) for row in selected]
    fig, axes = plt.subplots(1, 2, figsize=(max(9, len(selected) * 1.25), 4.3))
    for ax, values, title in zip(axes, (aucs, tars), ("Mean ROC-AUC", "Mean TAR at FAR=5%")):
        colors = ["#2563eb" if index == 0 else "#94a3b8" for index in range(len(values))]
        ax.bar(range(len(values)), values, color=colors)
        ax.set_ylim(max(0, min(values) - .08), min(1.01, max(values) + .04))
        ax.set_title(title); ax.grid(axis="y", alpha=.25)
        ax.set_xticks(range(len(values)), labels, rotation=35, ha="right")
        for index, value in enumerate(values):
            ax.text(index, value + .006, f"{value:.3f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(output, dpi=220, bbox_inches="tight"); plt.close(fig)


def save_multiseed(statistics_report: dict, output: Path) -> None:
    rows = []
    for name, metrics in statistics_report["multiseed_bootstrap"].items():
        rows.append((name, metrics["auc"]["mean"], metrics["auc"]["ci95"],
                     metrics["frr"]["mean"], metrics["frr"]["ci95"]))
    rows.sort(key=lambda item: item[0])
    labels = [row[0].replace("_", " ") for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, max(4.5, len(rows) * .38)))
    for ax, metric_index, ci_index, title in ((axes[0], 1, 2, "ROC-AUC (bootstrap 95% CI)"),
                                               (axes[1], 3, 4, "FRR at FAR=5% (bootstrap 95% CI)")):
        values = [row[metric_index] for row in rows]
        cis = [row[ci_index] for row in rows]
        errors = [[value-ci[0] for value, ci in zip(values, cis)],
                  [ci[1]-value for value, ci in zip(values, cis)]]
        axes_index = range(len(rows))
        ax.errorbar(values, axes_index, xerr=errors, fmt="o", capsize=3, color="#2563eb")
        ax.set_yticks(list(axes_index), labels if ax is axes[0] else [])
        ax.invert_yaxis(); ax.set_title(title); ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); fig.savefig(output, dpi=220, bbox_inches="tight"); plt.close(fig)


def save_scaling(efficiency: dict, output: Path) -> None:
    rows = sorted(efficiency["scaling"], key=lambda row: (row["nodes"], row["edges"]))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    axes[0].scatter([row["nodes"] for row in rows], [row["inference_ms"] for row in rows],
                    alpha=.65, color="#2563eb", s=22)
    axes[0].set(xlabel="Nodes", ylabel="R-GCN inference (ms)", title="Inference scaling")
    axes[1].scatter([row["edges"] for row in rows], [row["tensor_bytes"]/1024 for row in rows],
                    alpha=.65, color="#059669", s=22)
    axes[1].set(xlabel="Edges", ylabel="Graph tensors (KiB)", title="Memory scaling")
    for ax in axes: ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output, dpi=220, bbox_inches="tight"); plt.close(fig)


def write_efficiency_csv(efficiency: dict, target: Path) -> None:
    fields = ("method", "trainable_parameters", "checkpoint_mib", "tensorize_ms_mean",
              "inference_ms_mean", "batch_64_seconds", "models_per_second", "peak_gpu_mib")
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for name, model in efficiency["models"].items():
            writer.writerow({"method": name, "trainable_parameters": model["trainable_parameters"],
                             "checkpoint_mib": model["checkpoint_bytes"] / 2**20,
                             "tensorize_ms_mean": model["tensorize_ms_mean"],
                             "inference_ms_mean": model["inference_ms_mean"],
                             "batch_64_seconds": model["batch_64_seconds"],
                             "models_per_second": model["models_per_second"],
                             "peak_gpu_mib": model["peak_gpu_memory_bytes"] / 2**20})


def write_report(rows: list[dict], stats: dict, efficiency: dict, download: dict | None, output: Path) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows: groups[row["group"]].append(row)
    reference = next(row for row in rows if row["id"] == "full_seed2026")
    best_graph = max(groups["graph"], key=lambda row: float(row["auc_mean"]))
    training = [row for row in efficiency.get("training_runs", []) if row.get("training_seconds")]
    train_mean = sum(row["training_seconds"] for row in training) / len(training) if training else float("nan")
    peak_gpu = max((row.get("peak_gpu_memory_bytes") or 0 for row in training), default=0) / 2**30
    peak_rss = max((row.get("peak_process_rss_bytes") or 0 for row in training), default=0) / 2**30
    lines = ["# P1 mechanism and statistical experiment report", "",
             f"- Completed experiment entries: {len(rows)}/24.",
             f"- Full-model mean core AUC: {float(reference['auc_mean']):.4f}; mean TAR@FAR=5%: {float(reference['tar_far5_mean']):.4f}.",
             f"- Best graph ablation by mean AUC: {LABELS.get(best_graph['id'], best_graph['id'])} ({float(best_graph['auc_mean']):.4f}).",
             f"- Bootstrap resamples per attack: {stats['bootstrap_iterations']}; seeds: {', '.join(stats['seed_ids'])}.",
             f"- Mean measured training-only duration: {train_mean:.2f} s over {len(training)} newly measured runs.", "",
             "## Separated pipeline costs", "",
             f"- Training peak GPU allocation: {peak_gpu:.2f} GiB; process peak RSS: {peak_rss:.2f} GiB.",
             f"- XML parsing: {efficiency['xml_parse_ms']['mean']:.3f} ms/model (mean).",
             f"- Graph construction estimate: {efficiency['graph_construction_estimate_ms']['mean']:.3f} ms/model (mean)."]
    if download:
        lines.append(f"- Uncached dataset download: {download['bytes']/2**30:.2f} GiB in "
                     f"{download['elapsed_seconds']:.2f} s ({download['protocol']}).")
    for name, model in efficiency["models"].items():
        lines.append(f"- {name}: {model['trainable_parameters']:,} parameters, "
                     f"{model['inference_ms_mean']:.3f} ms/model inference, "
                     f"{model['models_per_second']:.1f} models/s for the measured 64-model pass.")
    for name, runtime in efficiency.get("traditional_runtime_ms", {}).items():
        lines.append(f"- {name}: {runtime:.3f} ms/model end-to-end handcrafted fingerprint extraction.")
    if efficiency.get("personalization"):
        lines.append(f"- Personalized registration: {efficiency['personalization']['elapsed_seconds']:.3f} s "
                     "for the 64-model registry plus validation background.")
    lines.extend(["", "The numerical source of every plot and table is retained in the generated CSV/JSON files. "
                  "Ablation conclusions follow the measured ranking: typed relation propagation is supported for "
                  "fixed-threshold TAR, where it exceeds no-edge and untyped alternatives, but not as a universal "
                  "ROC-AUC improvement.", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--efficiency", required=True)
    parser.add_argument("--download")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    stats = read_json(Path(args.statistics)); efficiency = read_json(Path(args.efficiency))
    download = read_json(Path(args.download)) if args.download else None
    rows = read_rows(Path(args.statistics).with_name("p1_ablation_summary.csv"))
    for group in ("loss", "feature", "graph"):
        save_ablation(rows, group, output / f"p1_{group}_ablation.png")
    save_multiseed(stats, output / "p1_multiseed_bootstrap.png")
    save_scaling(efficiency, output / "p1_efficiency_scaling.png")
    write_efficiency_csv(efficiency, output / "p1_efficiency.csv")
    write_report(rows, stats, efficiency, download, output / "P1_COMPLETION_REPORT.md")
    print(json.dumps({"output": str(output), "experiments": len(rows), "figures": 5}))


if __name__ == "__main__":
    main()
