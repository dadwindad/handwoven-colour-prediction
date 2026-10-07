"""CNN models on photos extracted from the source report (scripts/extract_images.py).

Zhu et al. (Color Res. Appl., CNN for silk yarn-dyed interwoven fabric) is not
open access, so its exact architecture is unknown. These models follow the
standard practice for small image datasets: an ImageNet-pretrained ResNet-18
(transfer learning). Two tasks:

  CNN-A  yarn photos -> fabric L*a*b*   same task as every model in models.py,
                                         so it joins the main comparison
  CNN-B  fabric photo -> fabric L*a*b*   reading the colour of an already woven
                                         fabric from an uncalibrated photo; a
                                         different task, reported separately

Photos in the report were not taken under controlled light, so they carry
camera and lighting error that the spectrophotometer values do not.
"""
import csv
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from .data import ROOT

IMAGES = ROOT / "data" / "images"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def _index():
    with open(IMAGES / "index.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load(path):
    return Image.open(ROOT / path).convert("RGB")


def _backbone():
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    return net.eval().to(DEVICE)


# ---------------------------------------------------------------- CNN-A
@lru_cache(maxsize=1)
def yarn_embeddings():
    """512-d ResNet-18 embedding per yarn code, averaged over all its photos and 4 crops."""
    net = _backbone()
    tf = transforms.Compose([transforms.Resize(256), transforms.FiveCrop(224)])
    to_t = transforms.Compose([transforms.ToTensor(), NORM])
    per_code = {}
    with torch.no_grad():
        for r in _index():
            if r["kind"] != "yarn":
                continue
            code = r["warp"] or r["weft"]
            crops = torch.stack([to_t(c) for c in tf(_load(r["file"]))]).to(DEVICE)
            per_code.setdefault(code, []).append(net(crops).mean(0).cpu().numpy())
    return {c: np.mean(v, axis=0) for c, v in per_code.items()}


EMBED_CACHE = IMAGES / "yarn_embeddings_resnet18.npz"


def build_yarn_embeddings():
    """Write the CNN-A feature cache. Run in its own process (see models.CNNYarnPhotos):
    loading PyTorch inside the same process as XGBoost/scikit-learn workers can
    deadlock on macOS (two OpenMP runtimes)."""
    emb = yarn_embeddings()
    np.savez(EMBED_CACHE, codes=np.array(list(emb)), emb=np.stack(list(emb.values())))
    return EMBED_CACHE


if __name__ == "__main__":
    print("wrote", build_yarn_embeddings())


# ---------------------------------------------------------------- CNN-B
class _FabricSet(torch.utils.data.Dataset):
    def __init__(self, paths, targets, train, size=192):
        self.paths, self.targets = paths, targets
        geo = [transforms.RandomResizedCrop(size, scale=(0.4, 1.0), ratio=(0.8, 1.25)),
               transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()] if train else \
              [transforms.Resize(size), transforms.CenterCrop(size)]
        # geometric augmentation only: colour jitter would corrupt the target
        self.tf = transforms.Compose(geo + [transforms.ToTensor(), NORM])
        self.images = [_load(p) for p in paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(self.images[i]), torch.tensor(self.targets[i] / 100.0, dtype=torch.float32)


def fabric_photo_paths(ds):
    by_pair = {(r["warp"], r["weft"]): r["file"] for r in _index() if r["kind"] == "fabric"}
    return [by_pair[(w, f)] for w, f in zip(ds.warp, ds.weft)]


def train_predict_cnn_b(paths, y, train_idx, test_idx, epochs=20, seed=0):
    """Fine-tune ResNet-18 on fabric photos of train_idx; return predicted Lab for test_idx."""
    torch.manual_seed(seed)
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(net.fc.in_features, 3)
    net = net.to(DEVICE)
    tr = _FabricSet([paths[i] for i in train_idx], y[train_idx], train=True)
    te = _FabricSet([paths[i] for i in test_idx], y[test_idx], train=False)
    dl = torch.utils.data.DataLoader(tr, batch_size=32, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        net.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            opt.step()
        sched.step()
    net.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in torch.utils.data.DataLoader(te, batch_size=64):
            # test-time augmentation: average over flips
            out = sum(net(t.to(DEVICE)) for t in (xb, xb.flip(-1), xb.flip(-2))) / 3
            preds.append(out.cpu().numpy())
    return np.vstack(preds) * 100.0


def photo_mean_lab(paths):
    """No-learning baseline: average colour of the central part of each photo, sRGB -> L*a*b* (D65)."""
    from .colorlib import xyz_to_lab
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    out = []
    for p in paths:
        a = np.asarray(_load(p), dtype=float) / 255
        h, w, _ = a.shape
        a = a[h // 6: -h // 6, w // 6: -w // 6]
        lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4).reshape(-1, 3).mean(0)
        out.append(xyz_to_lab(M @ lin * 100, "D65_2"))
    return np.array(out)
