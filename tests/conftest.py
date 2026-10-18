import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest
import torch

torch.set_num_threads(1)


@pytest.fixture
def small_cfg(tmp_path):
    """A config factory: offline shapes data, narrow networks, a handful of steps."""
    from gans.config import TrainConfig

    def make(method, **kw):
        defaults = {"method": method, "n_train": 256, "steps": 3, "batch_size": 16, "width": 8, "z_dim": 8,
                        "log_every": 1, "sample_every": 0, "output_dir": str(tmp_path / method)}
        if method not in ("pix2pix", "cyclegan"):
            defaults["dataset"] = "shapes"
        defaults.update(kw)
        return TrainConfig(**defaults)

    return make
