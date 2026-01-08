import argparse
import random
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
    ap.add_argument("--out_dir", type=str, default="data", help="Output folder containing imgs/ and masks/")
    ap.add_argument("--center", type=float, default=40.0)
    ap.add_argument("--width", type=float, default=400.0)
    ap.add_argument("--keep_empty_ratio", type=float, default=0.2,
                    help="Keep this fraction of empty slices (no spleen). 0.0~1.0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    task_dir = Path(args.msd_task_dir)
    img_tr = task_dir / "imagesTr"
    lab_tr = task_dir / "labelsTr"

    out_dir = Path(args.out_dir)
    out_imgs = out_dir / "imgs"
    out_masks = out_dir / "masks"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    img_files = sorted(img_tr.glob("*.nii.gz"))
    if not img_files:
        raise RuntimeError(f"No .nii.gz found in {img_tr}")

    total_saved = 0
    total_pos = 0
    total_neg = 0

    for img_path in tqdm(img_files, desc="Volumes"):
        name = img_path.name  # e.g. spleen_XX.nii.gz
        lab_path = lab_tr / name
        if not lab_path.exists():
            raise RuntimeError(f"Label not found for {img_path.name}: expected {lab_path}")

        img_nii = nib.load(str(img_path))
        lab_nii = nib.load(str(lab_path))

        img = img_nii.get_fdata(dtype=np.float32)  # shape (X,Y,Z)
        lab = lab_nii.get_fdata(dtype=np.float32)

        # Ensure integer labels
        lab = (lab > 0.5).astype(np.uint8)

        if img.shape != lab.shape:
            raise RuntimeError(f"Shape mismatch: {img_path.name} img {img.shape} vs lab {lab.shape}")

        case_id = img_path.name.replace(".nii.gz", "")

        # Axial slices along Z
        z_dim = img.shape[2]
        for z in range(z_dim):
            img2d = img[:, :, z]
            m2d = lab[:, :, z]

            has_fg = (m2d.max() > 0)
            if has_fg:
                total_pos += 1
            else:
                total_neg += 1
                if random.random() > args.keep_empty_ratio:
                    continue

            img_u8 = window_to_uint8(img2d, args.center, args.width)
            m_u8 = (m2d * 255).astype(np.uint8)

            stem = f"{case_id}_{z:04d}"
            Image.fromarray(img_u8).save(out_imgs / f"{stem}.png")
            Image.fromarray(m_u8).save(out_masks / f"{stem}_mask.png")
            total_saved += 1

    print(f"Done. saved={total_saved}, pos_slices={total_pos}, neg_slices_seen={total_neg}")
    print(f"Output: {out_imgs} and {out_masks}")


if __name__ == "__main__":
    main()
