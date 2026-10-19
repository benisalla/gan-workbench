"""Print the README results tables and draw the comparison figures from assets/.

    python scripts/results_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from gans.viz import compare_curves

ASSETS = Path("assets")


def fmt_p(n: int) -> str:
    return f"{n / 1e6:.2f} M"


def fmt_t(s: float) -> str:
    return f"{s / 60:.1f} min"


def labeled_table(rows: dict) -> str:
    lines = ["| method | G / D params | steps | time | FID (clf) ↓ | IS (clf) ↑ | modes ↑ | label acc |",
             "|---|---|---|---|---|---|---|---|"]
    real = rows.get("real")
    for name, r in rows.items():
        if name == "real":
            continue
        lines.append(f"| `{name}` | {fmt_p(r['params_g'])} / {fmt_p(r['params_d'])} | {r['steps']} | "
                     f"{fmt_t(r['seconds'])} | {r['fid']:.2f} | {r['is']:.2f} | {r['modes']}/{r['n_classes']} | "
                     f"{r['label_acc'] * 100:.1f}% |" if "label_acc" in r else
                     f"| `{name}` | {fmt_p(r['params_g'])} / {fmt_p(r['params_d'])} | {r['steps']} | "
                     f"{fmt_t(r['seconds'])} | {r['fid']:.2f} | {r['is']:.2f} | {r['modes']}/{r['n_classes']} | - |")
    if real:
        lines.append(f"| *real train images* | - | - | - | {real['fid']:.2f} | {real['is']:.2f} | "
                     f"{real['modes']}/{real['n_classes']} | - |")
    return "\n".join(lines)


def translation_table(rows: dict) -> str:
    lines = ["| method | G / D params | steps | time | L1 ↓ | mask IoU ↑ | cycle L1 ↓ |", "|---|---|---|---|---|---|---|"]
    for name, r in rows.items():
        cyc = f"{r['cycle_l1']:.3f}" if "cycle_l1" in r else "-"
        lines.append(f"| `{name}` | {fmt_p(r['params_g'])} / {fmt_p(r['params_d'])} | {r['steps']} | "
                     f"{fmt_t(r['seconds'])} | {r['l1']:.3f} | {r['iou']:.3f} | {cyc} |")
    return "\n".join(lines)


def sample_mosaic(ds: str, names: list[str], path: Path, cols: int = 4) -> None:
    rows_n = (len(names) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3.2, rows_n * 3.4))
    for ax in axes.flat:
        ax.axis("off")
    for ax, name in zip(axes.flat, names):
        f = ASSETS / f"samples_{ds}_{name}.png"
        if f.exists():
            ax.imshow(mpimg.imread(f))
            ax.set_title(name, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    for res in sorted(ASSETS.glob("results_*.json")):
        ds = res.stem.removeprefix("results_")
        rows = json.loads(res.read_text())
        print(f"\n### {ds}\n")
        labeled = "fid" in next(iter(rows.values()))
        print(labeled_table(rows) if labeled else translation_table(rows))
        names = [n for n in rows if n != "real"]
        if labeled:
            sample_mosaic(ds, names, ASSETS / f"mosaic_{ds}.png")
        hist = {n: json.loads((ASSETS / "histories" / f"{ds}_{n}.json").read_text()) for n in names
                if (ASSETS / "histories" / f"{ds}_{n}.json").exists()}
        if len(hist) > 1:
            compare_curves(hist, "d_fake", ASSETS / f"compare_dfake_{ds}.png", "D(G(z)) during training (smoothed)")
            compare_curves(hist, "update_g", ASSETS / f"compare_update_g_{ds}.png",
                           "generator update ratio ||dθ||/||θ||", logy=True)


if __name__ == "__main__":
    main()
