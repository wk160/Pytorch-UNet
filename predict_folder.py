import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from unet import UNet
from utils.data_loading import BasicDataset


def predict_one(net, pil_img, device, scale, threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, pil_img, scale, is_mask=False))
    img = img.unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        out = net(img)
        out = F.interpolate(out, (pil_img.size[1], pil_img.size[0]), mode="bilinear", align_corners=False)
        if net.n_classes > 1:
            mask = out.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)  # 0/1
        else:
            mask = (torch.sigmoid(out)[0, 0] > threshold).cpu().numpy().astype(np.uint8)

    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--classes", type=int, default=2)
    ap.add_argument("--bilinear", action="store_true", default=False)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear).to(device)
    state = torch.load(args.model, map_location=device)
    mask_values = state.pop("mask_values", [0, 255])
    net.load_state_dict(state)

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted([p for p in in_dir.glob("*.png") if not p.name.startswith("._")])

    for p in tqdm(img_files, desc="Predicting"):
        img = Image.open(p).convert("L")
        mask01 = predict_one(net, img, device, args.scale)

        # 保存成 0/255 PNG（和你训练 mask 一致）
        mask_u8 = (mask01 * 255).astype(np.uint8)
        Image.fromarray(mask_u8).save(out_dir / f"{p.stem}_mask.png")

    print("Done. Saved to:", out_dir)


if __name__ == "__main__":
    main()
