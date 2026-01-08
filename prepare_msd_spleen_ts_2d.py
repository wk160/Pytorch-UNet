import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm


def window_to_uint8(img2d: np.ndarray, center: float = 40.0, width: float = 400.0) -> np.ndarray:
    """CT windowing -> uint8 [0,255]."""
    img2d = img2d.astype(np.float32)
    lo = center - width / 2.0
    hi = center + width / 2.0
    img2d = np.clip(img2d, lo, hi)
    img2d = (img2d - lo) / (hi - lo + 1e-6)
    img2d = (img2d * 255.0).round().astype(np.uint8)
    return img2d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msd_task_dir", type=str, required=True, help="Path to extracted Task09_Spleen folder")
    ap.add_argument("--out_dir", type=str, default="data_ts", help="Output folder containing imgs/")
    ap.add_argument("--center", type=float, default=40.0, help="CT window center")
    ap.add_argument("--width", type=float, default=400.0, help="CT window width")
    ap.add_argument("--start_slice", type=int, default=0, help="Start slice index (inclusive)")
    ap.add_argument("--end_slice", type=int, default=-1, help="End slice index (exclusive). -1 means to the end.")
    args = ap.parse_args()

    task_dir = Path(args.msd_task_dir)
    img_ts = task_dir / "imagesTs"
    if not img_ts.exists():
        raise RuntimeError(f"Not found: {img_ts}")

    out_dir = Path(args.out_dir)
    out_imgs = out_dir / "imgs"
    out_imgs.mkdir(parents=True, exist_ok=True)

    # 过滤掉 macOS 的 AppleDouble 文件：._xxxx.nii.gz
    img_files = sorted([
        p for p in img_ts.glob("*.nii.gz")
        if not p.name.startswith("._") and not p.name.startswith(".")
    ])
    if not img_files:
        raise RuntimeError(f"No .nii.gz found in {img_ts}")

    saved = 0

    for img_path in tqdm(img_files, desc="Volumes (imagesTs)"):
        case_id = img_path.name.replace(".nii.gz", "")

        nii = nib.load(str(img_path))
        vol = nii.get_fdata(dtype=np.float32)  # typically (X,Y,Z)

        if vol.ndim != 3:
            raise RuntimeError(f"Unexpected ndim={vol.ndim} for {img_path.name}, expected 3D volume")

        z_dim = vol.shape[2]
        z_start = max(0, args.start_slice)
        z_end = z_dim if args.end_slice == -1 else min(z_dim, args.end_slice)

        for z in range(z_start, z_end):
            img2d = vol[:, :, z]
            img_u8 = window_to_uint8(img2d, args.center, args.width)

            stem = f"{case_id}_{z:04d}"
            Image.fromarray(img_u8).save(out_imgs / f"{stem}.png")
            saved += 1

    print(f"Done. Saved {saved} png slices to: {out_imgs}")


if __name__ == "__main__":
    main()
