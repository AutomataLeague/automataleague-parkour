"""Plot learning curves from the `metrics.jsonl` that every training run writes.

Seeded comparison in the usual RL-paper form: one panel per metric, algorithms as
series, and a shaded confidence band across seeds computed by seaborn from long-form
rows. Run several seeds per configuration and the band tells you whether a gap
between two algorithms is real; run one and it does not, which is why the seed count
is printed in every title.

    # everything under checkpoints/, grouped by run-name prefix
    uv run python tools/plot_curves.py --out renders/curves

    # explicit runs, labelled: <label>=<glob>
    uv run python tools/plot_curves.py --out renders/curves \
        PPO='checkpoints/ppo_flat_s*' SAC='checkpoints/sac_flat_s*'

Run names of the form `<algo>_<setting>_s<seed>` are grouped automatically, so a seed
sweep needs no arguments.
"""
import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set()

METRICS = [
    ("train/reward", "Episode Return", "return"),
    ("train/success_rate", "Success Rate", "success"),
    ("train/episode_length", "Episode Length (steps)", "eplen"),
    ("eval/reward", "Eval Return", "evalreturn"),
]
RUN_RE = re.compile(r"^(?P<algo>[a-z0-9]+)_(?P<setting>[a-z0-9]+)_s(?P<seed>\d+)$")


def load_run(metrics_path, key):
    """(frames, value) pairs for `key`, skipping rows that never logged it."""
    out = []
    with open(metrics_path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                      # a run killed mid-write leaves a partial line
            v = r.get(key)
            if v is not None and v == v:      # not NaN
                out.append((r["frames"], v))
    return out


def discover(root):
    """{(label, setting): [(seed, metrics_path)]} from `<algo>_<setting>_s<seed>` dirs."""
    groups = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "metrics.jsonl"))):
        name = os.path.basename(os.path.dirname(path))
        m = RUN_RE.match(name)
        if m:
            key = (m["algo"].upper(), m["setting"])
            groups.setdefault(key, []).append((int(m["seed"]), path))
        else:
            groups.setdefault((name, ""), []).append((0, path))
    return groups


def frame(groups, key, ylabel, bins):
    rows = []
    for (label, setting), entries in groups.items():
        for seed, path in entries:
            for frames, value in load_run(path, key):
                rows.append((setting, label, seed, frames, value))
    df = pd.DataFrame(rows, columns=["setting", "Algorithm", "seed",
                                     "Environment Steps", ylabel])
    if df.empty:
        return df
    # Bin the x axis so seeds logging at different cadences aggregate into the same
    # points; without this seaborn has one sample per x and draws no band.
    import numpy as np
    edges = np.linspace(0, df["Environment Steps"].max(), bins)
    idx = np.clip(np.digitize(df["Environment Steps"], edges) - 1, 0, len(edges) - 1)
    df["Environment Steps"] = edges[idx]
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", metavar="LABEL=GLOB",
                    help="explicit run groups; omit to auto-discover under --root")
    ap.add_argument("--root", default="checkpoints")
    ap.add_argument("--out", default="renders/curves")
    ap.add_argument("--bins", type=int, default=160, help="x-axis bins for aggregation")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    if args.runs:
        groups = {}
        for spec in args.runs:
            label, _, pattern = spec.partition("=")
            for i, d in enumerate(sorted(glob.glob(pattern))):
                p = os.path.join(d, "metrics.jsonl")
                if os.path.exists(p):
                    groups.setdefault((label, ""), []).append((i, p))
    else:
        groups = discover(args.root)
    if not groups:
        raise SystemExit(f"no metrics.jsonl found under {args.root}/*/")

    os.makedirs(args.out, exist_ok=True)
    settings = sorted({s for _, s in groups})
    written = 0
    for key, ylabel, slug in METRICS:
        df_all = frame(groups, key, ylabel, args.bins)
        if df_all.empty:
            continue
        for setting in settings:
            d = df_all[df_all.setting == setting]
            if d.empty:
                continue
            nseeds = d.groupby("Algorithm")["seed"].nunique()
            plt.figure(figsize=(15, 8))
            plt.ticklabel_format(style="sci", axis="x", useOffset=False, scilimits=(0, 0))
            ax = sns.lineplot(x="Environment Steps", y=ylabel, data=d,
                              hue="Algorithm", style="Algorithm", errorbar=("ci", 95))
            seeds = ", ".join(f"{a}: {n} seed{'s' if n != 1 else ''}"
                              for a, n in nseeds.items())
            name = f"parkour-1 {setting}".strip()
            plt.title(f"{name}   [{seeds}]", fontsize=15)
            ax.yaxis.label.set_size(15)
            ax.xaxis.label.set_size(15)
            plt.legend(loc="lower right", fontsize=15)
            tag = f"{slug}_{setting}" if setting else slug
            path = os.path.join(args.out, f"curve_{tag}.png")
            plt.savefig(path, dpi=args.dpi, bbox_inches="tight")
            plt.close()
            print(f"wrote {path}")
            written += 1
    if not written:
        raise SystemExit("metrics.jsonl found but none of the expected keys were logged")


if __name__ == "__main__":
    main()
