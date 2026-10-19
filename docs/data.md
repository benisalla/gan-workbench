# Data

[`data.py`](../src/gans/data.py). Every dataset is a pair of in-memory tensors scaled to
[-1, 1] (the range of the generators' tanh). Images are small, so there is no PIL transform
pipeline and no worker processes: a `TensorDataset` + `DataLoader(shuffle=True, drop_last=True)`.

| name | `gans-data` | size | content | used by |
|---|---|---|---|---|
| `mnist` | download 11 MB | 60k / 10k, 1×28×28, 10 labels | digits | all noise GANs (results) |
| `fashion` | download 30 MB | 60k / 10k, 1×28×28, 10 labels | clothes | same, harder |
| `cifar10` | download 170 MB | 50k / 10k, 3×32×32, 10 labels | colour photos | the GPU notebook |
| `shapes` | generated | 12k / 2k, 1×28×28, 4 labels | one white circle / square / triangle / cross, random size and position | offline tests and examples |
| `paired-shapes` | generated | 4k / 256 pairs, 1×32×32 | input: 1-3 shapes with random brightness gradients on a noisy, sloped background; target: their binary mask | `pix2pix` |
| `unpaired-shapes` | generated | 4k A + 4k B, 1×32×32 | A: filled shapes; B: 2-px outlines of *other*, independently drawn shapes; test keeps the true outline of each A | `cyclegan` |

```bash
gans-data mnist        # downloads to data/ and writes data/previews/mnist.png
gans-data paired-shapes
```

`--n-train N` keeps a random subset of the training split (for MNIST) or sets the number of
generated images (synthetic sets). Synthetic sets are deterministic for a given `--seed`.

**Why synthetic sets.** Everything must run offline in minutes on a CPU, and the image-to-image
methods need data where the right answer is known exactly: the mask of a shape, the outline of
a filled shape. That turns "does pix2pix / CycleGAN work?" into a number (IoU) instead of a
judgment call. The coursework's CMP Facades data (256 px, 606 pairs) is used in the notebook,
on a GPU.

## Facades (notebook only)

The notebook downloads CMP Facades with `kagglehub.dataset_download("kokeyehya/cmp-facade-db-base")`
(as the original coursework did), resizes to 128 or 256 px and trains `Pix2Pix` on it. The
U-Net's `depth` adapts to the size (depth 7 at 256 px gives the coursework's 1×1 bottleneck).
