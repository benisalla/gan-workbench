"""05 - pix2pix: noisy photo of shapes -> binary mask, U-Net + PatchGAN + L1 (about 3 minutes).

The coursework notebook did this on CMP Facades at 256 px; here the same architecture
(depth-4 U-Net, 6x6 PatchGAN) runs on 32 px synthetic pairs so it trains on a CPU.

    python examples/05_pix2pix_photo_to_mask.py
"""

from gans.config import TrainConfig
from gans.evaluate import evaluate_method
from gans.train import train
from gans.viz import save_translation_grid

r = train(TrainConfig(method="pix2pix", steps=600, log_every=20, sample_every=0,
                      output_dir="examples/output/05_pix2pix"))
method, data = r["model"], r["data"]
src, tgt = data.test
save_translation_grid(method, src[:12], "examples/output/05_pix2pix/test_pairs.png", target=tgt[:12])
m = evaluate_method(method, data)
print(f"test L1 {m['l1']:.3f}  mask IoU {m['iou']:.3f}")
print("rows: input / true mask / predicted mask -> examples/output/05_pix2pix/test_pairs.png")
