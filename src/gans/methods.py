"""The GAN types: a table of specs plus three small classes that do one D step and one G step.

A GAN type here is just a choice of (networks, loss, conditioning, Lipschitz control, optimiser).
``METHODS`` lists them; ``build_method`` turns a spec + config into an object with

* ``d_step(batch) -> dict``  one discriminator / critic update
* ``g_step(batch) -> dict``  one generator update
* ``sample(n, y)``           images from the generator (EMA copy if enabled)

and the one training loop in ``train.py`` calls them for every type. Three classes cover all ten:

* ``NoiseGAN``  z -> image (gan, dcgan, cgan, acgan, lsgan, wgan, wgan-gp, sngan)
* ``Pix2Pix``   paired image -> image, conditional PatchGAN + L1
* ``CycleGAN``  unpaired A <-> B, two generators, two PatchGANs, cycle-consistency L1
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from gans.config import TrainConfig
from gans.data import DataInfo
from gans.losses import d_loss, feature_matching, g_loss, gradient_penalty, r1_penalty
from gans.models import (
    DCDiscriminator,
    DCGenerator,
    MLPDiscriminator,
    MLPGenerator,
    PatchDiscriminator,
    UNetGenerator,
)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str            # "noise" | "pix2pix" | "cyclegan"
    arch: str              # "mlp" | "dcgan" | "unet"
    loss: str              # see losses.LOSSES
    summary: str
    conditional: str = "none"   # "none" | "input" (cGAN) | "aux" (ACGAN)
    d_norm: str = "batch"       # "batch" | "none" | "spectral"
    n_critic: int = 1
    clip: float = 0.0           # WGAN weight clipping
    gp: float = 0.0             # WGAN-GP penalty weight
    optimizer: str = "adam"     # "adam" | "rmsprop"
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    betas: tuple[float, float] = (0.5, 0.999)
    dataset: str = "mnist"


METHODS: dict[str, MethodSpec] = {s.name: s for s in [
    MethodSpec("gan", "noise", "mlp", "bce",
               "MLPs on flat pixels, D is a real/fake classifier (Goodfellow 2014)"),
    MethodSpec("dcgan", "noise", "dcgan", "bce",
               "same loss, strided (transposed) convolutions + BatchNorm (Radford 2015)"),
    MethodSpec("cgan", "noise", "dcgan", "bce",
               "label fed to G (with z) and to D (as channel maps): p(x | y) (Mirza 2014)",
               conditional="input"),
    MethodSpec("acgan", "noise", "dcgan", "bce",
               "label fed to G; D predicts real/fake AND the class (Odena 2017)", conditional="aux"),
    MethodSpec("lsgan", "noise", "dcgan", "ls",
               "least-squares instead of log loss: gradients do not vanish for confident D (Mao 2017)"),
    MethodSpec("wgan", "noise", "dcgan", "wgan",
               "critic estimates the Wasserstein-1 distance, Lipschitz by weight clipping (Arjovsky 2017)",
               d_norm="none", n_critic=5, clip=0.01, optimizer="rmsprop", lr_g=5e-5, lr_d=5e-5),
    MethodSpec("wgan-gp", "noise", "dcgan", "wgan",
               "Wasserstein critic, Lipschitz by a gradient penalty on interpolates (Gulrajani 2017)",
               d_norm="none", n_critic=5, gp=10.0, lr_g=1e-4, lr_d=1e-4, betas=(0.0, 0.9)),
    MethodSpec("sngan", "noise", "dcgan", "hinge",
               "hinge loss, every D layer spectrally normalised (Miyato 2018)",
               d_norm="spectral", betas=(0.0, 0.9)),
    MethodSpec("pix2pix", "pix2pix", "unet", "bce",
               "paired image -> image: U-Net G, conditional PatchGAN D, + 100 x L1 (Isola 2017)",
               dataset="paired-shapes"),
    MethodSpec("cyclegan", "cyclegan", "unet", "ls",
               "unpaired A <-> B: two G, two PatchGAN D, cycle loss ||G_BA(G_AB(a)) - a|| (Zhu 2017)",
               d_norm="instance", dataset="unpaired-shapes"),
]}
ALIASES = {"hinge": "sngan", "wgangp": "wgan-gp", "wgan_gp": "wgan-gp", "vanilla": "gan"}


def get_spec(name: str) -> MethodSpec:
    name = ALIASES.get(name, name)
    if name not in METHODS:
        raise ValueError(f"unknown method {name!r}, choose from {list(METHODS)}")
    return METHODS[name]


def resolve(cfg: TrainConfig) -> TrainConfig:
    """Fill the None fields of a config from its method's defaults (returns a copy)."""
    cfg = copy.deepcopy(cfg)
    spec = get_spec(cfg.method)
    cfg.method = spec.name
    cfg.dataset = cfg.dataset or spec.dataset
    cfg.n_critic = cfg.n_critic or spec.n_critic
    cfg.betas = tuple(cfg.betas) if cfg.betas else spec.betas
    if cfg.ttur and cfg.lr_g is None and cfg.lr_d is None:
        cfg.lr_g, cfg.lr_d = 1e-4, 4e-4
    cfg.lr_g = cfg.lr_g or spec.lr_g
    cfg.lr_d = cfg.lr_d or spec.lr_d
    cfg.output_dir = cfg.output_dir or f"checkpoints/{spec.name}"
    return cfg


def _optim(params, kind: str, lr: float, betas) -> torch.optim.Optimizer:
    if kind == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    return torch.optim.Adam(params, lr=lr, betas=betas)


def _score(logit: torch.Tensor, loss: str) -> float:
    """What D 'says' about a batch: a probability for BCE, the raw score otherwise."""
    return (torch.sigmoid(logit) if loss == "bce" else logit).mean().item()


class GANMethod:
    G: nn.Module
    D: nn.Module

    def __init__(self, spec: MethodSpec, cfg: TrainConfig, info: DataInfo):
        self.spec, self.cfg, self.info = spec, cfg, info
        self.device = torch.device(cfg.device)
        self.ema_G: nn.Module | None = None

    def _finish(self) -> None:
        self.G.to(self.device)
        self.D.to(self.device)
        self.opt_g = _optim(self.G.parameters(), self.spec.optimizer, self.cfg.lr_g, self.cfg.betas)
        self.opt_d = _optim(self.D.parameters(), self.spec.optimizer, self.cfg.lr_d, self.cfg.betas)

    def generator(self) -> nn.Module:
        """The network to sample from: the EMA copy of G when ``--ema`` is on."""
        return self.ema_G if self.ema_G is not None else self.G

    def _to(self, batch):
        return [t.to(self.device) for t in batch]


class NoiseGAN(GANMethod):
    """z (+ label) -> image. Covers gan, dcgan, cgan, acgan, lsgan, wgan, wgan-gp, sngan."""

    def __init__(self, spec: MethodSpec, cfg: TrainConfig, info: DataInfo):
        super().__init__(spec, cfg, info)
        k = info.n_classes if spec.conditional != "none" else 0
        if spec.conditional != "none" and k == 0:
            raise ValueError(f"{spec.name} needs a labeled dataset")
        self.n_classes = k
        d_cond = k if spec.conditional == "input" else 0        # cGAN: D sees the label
        aux = k if spec.conditional == "aux" else 0             # ACGAN: D predicts the label
        c, s, w = info.channels, info.size, cfg.width
        if spec.arch == "mlp":
            self.G = MLPGenerator(cfg.z_dim, c, s, 4 * w, k)
            self.D = MLPDiscriminator(c, s, 4 * w, d_cond, aux, spectral=spec.d_norm == "spectral")
        else:
            self.G = DCGenerator(cfg.z_dim, c, s, w, k)
            self.D = DCDiscriminator(c, s, w, d_cond, aux, norm=spec.d_norm)
        self._finish()

    def noise(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.cfg.z_dim, device=self.device)

    def d_step(self, batch) -> dict:
        x, y = self._to(batch)
        spec, cfg = self.spec, self.cfg
        y_g = y if self.n_classes else None                       # fakes get the batch's labels
        y_d = y if spec.conditional == "input" else None
        with torch.no_grad():
            fake = self.G(self.noise(x.size(0)), y_g)
        r, f = self.D(x, y_d), self.D(fake, y_d)
        loss = d_loss(spec.loss, r.logit, f.logit, real_target=1.0 - cfg.label_smoothing)
        logs = {}
        if spec.conditional == "aux":
            cls = F.cross_entropy(r.class_logits, y) + F.cross_entropy(f.class_logits, y)
            loss = loss + cls
            logs["d_class_acc"] = (r.class_logits.argmax(1) == y).float().mean().item()
        if spec.gp:
            gp = gradient_penalty(self.D, x, fake, y_d)
            loss = loss + spec.gp * gp
            logs["gp"] = gp.item()
        if cfg.r1:
            loss = loss + cfg.r1 * r1_penalty(self.D, x, y_d)
        self.opt_d.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_d.step()
        if spec.clip:                                             # WGAN: keep weights in [-c, c]
            with torch.no_grad():
                for p in self.D.parameters():
                    p.clamp_(-spec.clip, spec.clip)
        return {"d_loss": loss.item(), "d_real": _score(r.logit, spec.loss),
                "d_fake": _score(f.logit, spec.loss), **logs}

    def g_step(self, batch) -> dict:
        x, y = self._to(batch)
        spec = self.spec
        y_g = y if self.n_classes else None
        y_d = y if spec.conditional == "input" else None
        fake = self.G(self.noise(x.size(0)), y_g)
        f = self.D(fake, y_d)
        loss = g_loss(spec.loss, f.logit, saturating=self.cfg.saturating)
        if spec.conditional == "aux":
            loss = loss + F.cross_entropy(f.class_logits, y)
        if self.cfg.feature_matching:
            with torch.no_grad():
                feat_real = self.D(x, y_d).features
            loss = loss + feature_matching(feat_real, f.features)
        self.opt_g.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_g.step()
        return {"g_loss": loss.item(), "d_fake_g": _score(f.logit, spec.loss)}

    @torch.no_grad()
    def sample(self, n: int, y: torch.Tensor | None = None, z: torch.Tensor | None = None) -> torch.Tensor:
        G = self.generator()
        was = G.training
        G.eval()
        z = self.noise(n) if z is None else z.to(self.device)
        if self.n_classes and y is None:
            y = torch.arange(n, device=self.device) % self.n_classes
        out = G(z, y.to(self.device) if y is not None else None)
        G.train(was)
        return out.cpu()


class Pix2Pix(GANMethod):
    """Paired translation src -> tgt: G is a U-Net, D sees (src, image) and judges 6x6 patches."""

    def __init__(self, spec: MethodSpec, cfg: TrainConfig, info: DataInfo):
        super().__init__(spec, cfg, info)
        c, w, depth = info.channels, cfg.width // 2, cfg.extra.get("unet_depth", 4)
        self.G = UNetGenerator(c, c, width=w, depth=depth)
        self.D = PatchDiscriminator(c, cond_ch=c, width=w, norm="batch")
        self._finish()

    def d_step(self, batch) -> dict:
        src, tgt = self._to(batch)
        with torch.no_grad():
            fake = self.G(src)
        r, f = self.D(tgt, src), self.D(fake, src)
        loss = 0.5 * d_loss("bce", r.logit, f.logit, 1.0 - self.cfg.label_smoothing)
        self.opt_d.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_d.step()
        return {"d_loss": loss.item(), "d_real": _score(r.logit, "bce"), "d_fake": _score(f.logit, "bce")}

    def g_step(self, batch) -> dict:
        src, tgt = self._to(batch)
        fake = self.G(src)
        adv = g_loss("bce", self.D(fake, src).logit)
        l1 = F.l1_loss(fake, tgt)
        loss = adv + self.cfg.lambda_l1 * l1
        self.opt_g.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_g.step()
        return {"g_loss": loss.item(), "g_adv": adv.item(), "l1": l1.item()}

    @torch.no_grad()
    def translate(self, src: torch.Tensor) -> torch.Tensor:
        G = self.generator()
        was = G.training
        G.eval()
        out = G(src.to(self.device)).cpu()
        G.train(was)
        return out


class CycleGAN(GANMethod):
    """Unpaired A <-> B. G = {ab, ba}, D = {a, b}; LSGAN loss + lambda * cycle L1 (+ identity)."""

    def __init__(self, spec: MethodSpec, cfg: TrainConfig, info: DataInfo):
        super().__init__(spec, cfg, info)
        c, w, depth = info.channels, cfg.width // 2, cfg.extra.get("unet_depth", 4)
        self.G = nn.ModuleDict({"ab": UNetGenerator(c, c, w, depth, dropout=False),
                                "ba": UNetGenerator(c, c, w, depth, dropout=False)})
        self.D = nn.ModuleDict({"a": PatchDiscriminator(c, 0, w, norm="instance"),
                                "b": PatchDiscriminator(c, 0, w, norm="instance")})
        self._finish()

    def d_step(self, batch) -> dict:
        a, b = self._to(batch)
        with torch.no_grad():
            fake_b, fake_a = self.G["ab"](a), self.G["ba"](b)
        rb, fb = self.D["b"](b).logit, self.D["b"](fake_b).logit
        ra, fa = self.D["a"](a).logit, self.D["a"](fake_a).logit
        loss = 0.5 * (d_loss("ls", rb, fb) + d_loss("ls", ra, fa))
        self.opt_d.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_d.step()
        return {"d_loss": loss.item(), "d_real": 0.5 * (rb.mean() + ra.mean()).item(),
                "d_fake": 0.5 * (fb.mean() + fa.mean()).item()}

    def g_step(self, batch) -> dict:
        a, b = self._to(batch)
        fake_b, fake_a = self.G["ab"](a), self.G["ba"](b)
        adv = g_loss("ls", self.D["b"](fake_b).logit) + g_loss("ls", self.D["a"](fake_a).logit)
        cycle = F.l1_loss(self.G["ba"](fake_b), a) + F.l1_loss(self.G["ab"](fake_a), b)
        loss = adv + self.cfg.lambda_cycle * cycle
        if self.cfg.lambda_identity:
            ident = F.l1_loss(self.G["ab"](b), b) + F.l1_loss(self.G["ba"](a), a)
            loss = loss + self.cfg.lambda_identity * ident
        self.opt_g.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_g.step()
        return {"g_loss": loss.item(), "g_adv": adv.item(), "cycle": cycle.item()}

    @torch.no_grad()
    def translate(self, x: torch.Tensor, direction: str = "ab") -> torch.Tensor:
        G = self.generator()[direction]
        was = G.training
        G.eval()
        out = G(x.to(self.device)).cpu()
        G.train(was)
        return out


FAMILIES = {"noise": NoiseGAN, "pix2pix": Pix2Pix, "cyclegan": CycleGAN}


def build_method(cfg: TrainConfig, info: DataInfo) -> GANMethod:
    cfg = resolve(cfg)
    spec = get_spec(cfg.method)
    return FAMILIES[spec.family](spec, cfg, info)
