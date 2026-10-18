"""Train and evaluate several GAN types on the same data with the same budget.

    python scripts/compare_all.py --methods gan dcgan cgan --dataset mnist --steps 3000

For every method: train with the shared loop (checkpoints/compare/<dataset>/<method>), evaluate
(classifier-feature FID, IS analogue, mode coverage, label accuracy; or L1 / IoU for the
image-to-image methods), copy the sample grid to assets/, and merge one row into
assets/results_<dataset>.json. Run it in several calls to keep each run short; then
``scripts/results_table.py`` prints the markdown table and draws the comparison figures.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from gans.config import TrainConfig
from gans.evaluate import classifier_for, evaluate_method, real_baseline
from gans.train import train
from gans.viz import interpolation_grid


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--n-eval", type=int, default=5000)
    p.add_argument("--tag", default="", help="suffix for the row name, e.g. '+ema'")
    p.add_argument("--extra", default="{}", help='JSON dict of extra TrainConfig fields, e.g. \'{"ema": 0.999}\'')
    p.add_argument("--assets", default="assets")
    args = p.parse_args()
    extra = json.loads(args.extra)
    assets = Path(args.assets)
    assets.mkdir(exist_ok=True)
    for name in args.methods:
        cfg = TrainConfig(method=name, dataset=args.dataset, steps=args.steps, sample_every=0,
                          **extra)
        row_name = name + args.tag
        ds = args.dataset or ("paired-shapes" if name == "pix2pix" else
                              "unpaired-shapes" if name == "cyclegan" else "mnist")
        cfg.output_dir = f"checkpoints/compare/{ds}/{row_name}"
        r = train(cfg)
        method, data = r["model"], r["data"]
        clf = classifier_for(ds, data=data) if data.info.kind == "labeled" else None
        metrics = evaluate_method(method, data, args.n_eval, clf=clf)
        row = {"method": row_name, "dataset": ds, "steps": args.steps, "seconds": round(r["seconds"], 1),
               "params_g": r["params_g"], "params_d": r["params_d"], **metrics}
        print(json.dumps(row))
        out = Path(cfg.output_dir)
        shutil.copy(out / "samples.png", assets / f"samples_{ds}_{row_name}.png")
        shutil.copy(out / "curves.png", assets / f"curves_{ds}_{row_name}.png")
        if data.info.kind == "labeled":
            interpolation_grid(method, path=assets / f"interp_{ds}_{row_name}.png")
        res_path = assets / f"results_{ds}.json"
        rows = json.loads(res_path.read_text()) if res_path.exists() else {}
        rows[row_name] = row
        if data.info.kind == "labeled" and "real" not in rows:
            rows["real"] = {"method": "real", "dataset": ds, **real_baseline(ds, n=args.n_eval)}
        res_path.write_text(json.dumps(rows, indent=2))
        (assets / "histories").mkdir(exist_ok=True)
        shutil.copy(out / "history.json", assets / "histories" / f"{ds}_{row_name}.json")


if __name__ == "__main__":
    main()
