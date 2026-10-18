import json

from gans import data as data_cli
from gans import evaluate as eval_cli
from gans import sample as sample_cli
from gans import train as train_cli


def test_cli_train_sample_eval_on_shapes(tmp_path, capsys):
    out = tmp_path / "run"
    train_cli.main(["--method", "acgan", "--dataset", "shapes", "--n-train", "128", "--steps", "3",
                    "--width", "8", "--z-dim", "8", "--batch-size", "16", "--log-every", "1",
                    "--output-dir", str(out)])
    ckpt = out / "last.pt"
    sample_cli.main([str(ckpt), "--n", "8", "--out", str(tmp_path / "s.png")])
    sample_cli.main([str(ckpt), "--interpolate", "--out", str(tmp_path / "i.png")])
    assert (tmp_path / "s.png").exists() and (tmp_path / "i.png").exists()
    import gans.evaluate as ev

    ev_cache = tmp_path / "cache"
    orig = ev.classifier_for
    ev.classifier_for = lambda d, data_dir="data", cache_dir="c", data=None: orig(d, data_dir, ev_cache, data)
    try:
        eval_cli.main([str(ckpt), "--n", "200"])
    finally:
        ev.classifier_for = orig
    metrics = json.loads((out / "metrics.json").read_text())
    assert {"fid", "is", "modes", "label_acc"} <= set(metrics)


def test_cli_data_writes_preview(tmp_path):
    data_cli.main(["paired-shapes", "--data-dir", str(tmp_path)])
    assert (tmp_path / "previews" / "paired-shapes.png").exists()
