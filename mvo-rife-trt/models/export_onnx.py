"""Export ONNX fp16 des modèles RIFE de mvo-rife-trt (TensorRT for RTX).

Architectures IFNet v4.6, v4.15 et v4.15-lite reprises de vs-rife (MIT, HolyWu ; Practical-RIFE, hzwer, MIT).
Interface identique aux modèles ncnn de muxiveo-rife : in0, in1 (RGB [0,1], fp16, NCHW), in2 (carte de
temps, fp16) → out0 (RGB fp16). Dimensions dynamiques (multiples du padding de muxiveo-rife).
Variante « -uhd » : échelles du flux doublées (équivalent de muxiveo-rife --uhd / Practical-RIFE scale=0.5).

Usage : python export_onnx.py <dossier des poids> <dossier de sortie>
Les poids (models.json) sont vérifiés par SHA-256 avant usage.
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent


def conv(cin, cout, k=3, s=1, p=1, d=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, dilation=d, bias=True), nn.LeakyReLU(0.2, True))


class Head(nn.Module):
    def __init__(self, c=32, out=8):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, c, 3, 2, 1)
        self.cnn1 = nn.Conv2d(c, c, 3, 1, 1)
        self.cnn2 = nn.Conv2d(c, c, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(c, out, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = x.clamp(0.0, 1.0)
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class ResConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, cin, c, outc=6):
        super().__init__()
        self.conv0 = nn.Sequential(conv(cin, c // 2, 3, 2, 1), conv(c // 2, c, 3, 2, 1))
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * outc, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x, flow, scale: float):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear")
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear") / scale
            x = torch.cat((x, flow), 1)
        feat = self.convblock(self.conv0(x))
        tmp = F.interpolate(self.lastconv(feat), scale_factor=scale, mode="bilinear")
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


def warp(x, flow):
    """Déformation arrière par flux en pixels (bilinéaire, bord répliqué), grille au type de x."""
    b, _, h, w = x.shape
    gx = torch.linspace(-1.0, 1.0, w, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    gy = torch.linspace(-1.0, 1.0, h, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
    g = torch.cat([gx + flow[:, 0:1] / ((w - 1.0) / 2.0), gy + flow[:, 1:2] / ((h - 1.0) / 2.0)], 1)
    return F.grid_sample(x, g.permute(0, 2, 3, 1).to(x.dtype), mode="bilinear", padding_mode="border",
                         align_corners=True)


class Net415(nn.Module):
    """IFNet v4.15 (encodeur de caractéristiques, masque remplacé à chaque étape) ; « lite » : canaux réduits."""

    def __init__(self, scales, lite=False):
        super().__init__()
        f = 8 if lite else 16                      # caractéristiques des deux images
        c = (128, 96, 64, 48) if lite else (192, 128, 96, 64)
        self.block0 = IFBlock(7 + f, c[0])
        self.block1 = IFBlock(8 + 4 + f, c[1])
        self.block2 = IFBlock(8 + 4 + f, c[2])
        self.block3 = IFBlock(8 + 4 + f, c[3])
        self.encode = Head(16, 4) if lite else Head()
        self.scales = scales

    def forward(self, in0, in1, in2):
        img0, img1, t = in0.clamp(0.0, 1.0), in1.clamp(0.0, 1.0), in2
        f0, f1 = self.encode(img0), self.encode(img1)
        blocks = [self.block0, self.block1, self.block2, self.block3]
        wi0, wi1, flow, mask = img0, img1, None, None
        for i in range(4):
            if flow is None:
                flow, mask, _ = blocks[i](torch.cat((img0, img1, f0, f1, t), 1), None, self.scales[i])
            else:
                wf0, wf1 = warp(f0, flow[:, :2]), warp(f1, flow[:, 2:4])
                fd, mask, _ = blocks[i](torch.cat((wi0, wi1, wf0, wf1, t, mask), 1), flow, self.scales[i])
                flow = flow + fd
            wi0, wi1 = warp(img0, flow[:, :2]), warp(img1, flow[:, 2:4])
        mask = torch.sigmoid(mask)
        return wi0 * mask + wi1 * (1 - mask)


class Net46(nn.Module):
    """IFNet v4.6 (sans encodeur, masque cumulé d'une étape à l'autre)."""

    def __init__(self, scales):
        super().__init__()
        self.block0 = IFBlock(7, 192)
        self.block1 = IFBlock(8 + 4, 128)
        self.block2 = IFBlock(8 + 4, 96)
        self.block3 = IFBlock(8 + 4, 64)
        self.scales = scales

    def forward(self, in0, in1, in2):
        img0, img1, t = in0.clamp(0.0, 1.0), in1.clamp(0.0, 1.0), in2
        blocks = [self.block0, self.block1, self.block2, self.block3]
        wi0, wi1, flow, mask = img0, img1, None, None
        for i in range(4):
            if flow is None:
                flow, mask, _ = blocks[i](torch.cat((img0, img1, t), 1), None, self.scales[i])
            else:
                fd, md, _ = blocks[i](torch.cat((wi0, wi1, t, mask), 1), flow, self.scales[i])
                flow, mask = flow + fd, mask + md
            wi0, wi1 = warp(img0, flow[:, :2]), warp(img1, flow[:, 2:4])
        mask = torch.sigmoid(mask)
        return wi0 * mask + wi1 * (1 - mask)


class Net425(nn.Module):
    """IFNet v4.25 : cinq étages (échelles 16 à 1), caractéristiques transmises d'un étage à l'autre ;
    « heavy » : canaux doublés."""

    def __init__(self, scales, heavy=False):
        super().__init__()
        k = 2 if heavy else 1
        c = (192 * k, 128 * k, 96 * k, 64 * k, 32 * k)
        self.block0 = IFBlock(7 + 8, c[0], outc=13)
        self.block1 = IFBlock(8 + 4 + 8 + 8, c[1], outc=13)
        self.block2 = IFBlock(8 + 4 + 8 + 8, c[2], outc=13)
        self.block3 = IFBlock(8 + 4 + 8 + 8, c[3], outc=13)
        self.block4 = IFBlock(8 + 4 + 8 + 8, c[4], outc=13)
        self.encode = Head(16, 4)
        self.scales = scales

    def forward(self, in0, in1, in2):
        img0, img1, t = in0.clamp(0.0, 1.0), in1.clamp(0.0, 1.0), in2
        f0, f1 = self.encode(img0), self.encode(img1)
        blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]
        wi0, wi1, flow, mask, feat = img0, img1, None, None, None
        for i in range(5):
            if flow is None:
                flow, mask, feat = blocks[i](torch.cat((img0, img1, f0, f1, t), 1), None, self.scales[i])
            else:
                wf0, wf1 = warp(f0, flow[:, :2]), warp(f1, flow[:, 2:4])
                fd, mask, feat = blocks[i](torch.cat((wi0, wi1, wf0, wf1, t, mask, feat), 1), flow, self.scales[i])
                flow = flow + fd
            wi0, wi1 = warp(img0, flow[:, :2]), warp(img1, flow[:, 2:4])
        mask = torch.sigmoid(mask)
        return wi0 * mask + wi1 * (1 - mask)


ARCHS = {
    "rife-v4.6": Net46,
    "rife-v4.15": Net415,
    "rife-v4.15-mvo1": Net415,              # RIFE v4.15 affiné par Muxiveo (même architecture)
    "rife-v4.15-lite": lambda scales: Net415(scales, lite=True),
    "rife-v4.25": Net425,
    "rife-v4.25-heavy": lambda scales: Net425(scales, heavy=True),
}
# Échelles des flux par étage (mode normal) ; « -uhd » : échelles doublées.
SCALES = {"rife-v4.25": [16.0, 8.0, 4.0, 2.0, 1.0], "rife-v4.25-heavy": [16.0, 8.0, 4.0, 2.0, 1.0]}


def fetch_weights(name: str, info: dict, weights_dir: Path) -> Path:
    path = weights_dir / Path(info["url"]).name
    if not path.is_file():
        weights_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(info["url"], timeout=120) as resp:  # nosec B310 — URL GitHub épinglée
            path.write_bytes(resp.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != info["sha256"]:
        raise SystemExit(f"{name} : SHA-256 inattendu ({digest})")
    return path


def export(name: str, weights: Path, uhd: bool, out_path: Path) -> None:
    scales = SCALES.get(name, [8.0, 4.0, 2.0, 1.0])
    if uhd:
        scales = [2 * s for s in scales]
    net = ARCHS[name](scales).eval()
    ckpt = torch.load(weights, map_location="cpu")
    state = {k.replace("module.", ""): v for k, v in ckpt.items() if "teacher" not in k and "caltime" not in k}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        raise SystemExit(f"{name} : poids manquants {missing}")
    net = net.half()
    size = 256
    args = (torch.rand(1, 3, size, size).half(), torch.rand(1, 3, size, size).half(), torch.full((1, 1, size, size), 0.5).half())
    torch.onnx.export(net, args, str(out_path), input_names=["in0", "in1", "in2"], output_names=["out0"],
                      opset_version=17, dynamo=False,
                      dynamic_axes={n: {2: "h", 3: "w"} for n in ("in0", "in1", "in2", "out0")})
    print(f"{out_path.name} : {out_path.stat().st_size / 1e6:.1f} Mo", flush=True)


def main() -> None:
    weights_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HERE / "models.json").read_text(encoding="utf-8"))
    for name, info in manifest["weights"].items():
        weights = fetch_weights(name, info, weights_dir)
        for variant in manifest["variants"]:
            export(name, weights, variant == "-uhd", out_dir / f"{name}{variant}.onnx")


if __name__ == "__main__":
    main()
