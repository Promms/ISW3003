import os
import glob
import numpy as np
from PIL import Image

from utils import make_panel, load_pair

BILINEAR = Image.Resampling.BILINEAR
NEAREST = Image.Resampling.NEAREST
AFFINE = Image.Transform.AFFINE
FLIP_LR = Image.Transpose.FLIP_LEFT_RIGHT


class SegAugmentation:
    """Jointly transform an image and its segmentation mask.

    - Image: BILINEAR interpolation.
    - Mask: NEAREST interpolation (preserves class indices).
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    # --- Reference implementation (already done) -----------------------------
    def rotate(self, img: Image.Image, mask: Image.Image, max_deg: float = 30.0):
        angle = float(self.rng.uniform(-max_deg, max_deg))
        img_out = img.rotate(angle, resample=BILINEAR, fillcolor=(0, 0, 0))
        mask_out = mask.rotate(angle, resample=NEAREST, fillcolor=0)
        return img_out, mask_out

    # --- TODO: implement the following augmentations ------------------------

    def random_crop(self, img: Image.Image, mask: Image.Image, size_frac=(0.5, 0.9)):
        # TODO: sample a crop size as a random fraction of the original size.
        raise NotImplementedError

    def flip(self, img: Image.Image, mask: Image.Image):
        # TODO: apply a horizontal flip to both the image and mask.
        #       Note: vertical flip is NOT used for semantic segmentation
        #       because natural scenes have a fixed gravity/orientation.
        raise NotImplementedError

    def blur(self, img: Image.Image, mask: Image.Image, sigma_range=(0.8, 3.0)):
        # TODO: sample `sigma` uniformly from `sigma_range`, then apply Gaussian blur.
        raise NotImplementedError

    def scale(self, img: Image.Image, mask: Image.Image, scale_range=(0.7, 1.3)):
        # TODO: sample a random scale factor and resize the image and mask.
        raise NotImplementedError

    def translate(self, img: Image.Image, mask: Image.Image, max_frac: float = 0.2):
        # TODO: sample per-axis shifts `tx`, `ty` as fractions of W, H and shift the image.
        raise NotImplementedError

    def partial_erase(self, img: Image.Image, mask: Image.Image, frac_range=(0.1, 0.35)):
        # TODO: pick a random rectangle (x, y, w, h) inside the image, then
        #       fill that region with the image's per-channel mean pixel on
        #       the image and with class 0 (background) on the mask.
        raise NotImplementedError

    # --- Two-image augmentations (not included in apply_random) -------------

    def stitch(self, img_a: Image.Image, mask_a: Image.Image,
               img_b: Image.Image, mask_b: Image.Image):
        # TODO: concatenate (A | B) horizontally.
        #       Resize B to match A's height (keeping B's aspect ratio) then paste
        #       both into a new Image.
        raise NotImplementedError

    def copy_paste(self, target_img: Image.Image, target_mask: Image.Image,
                   source_img: Image.Image, source_mask: Image.Image):
        # TODO: pick a random non-background class from `source_mask`.
        #       Compute its bounding box and crop the image/mask.
        #       If the cropped region is larger than the target, shrink it.
        #       Sample a random paste position inside the target, then paste.
        #       Do the same on `out_mask` with the class-index crop so the
        #       pasted pixels carry the correct class labels.
        raise NotImplementedError

    # --- Random composition --------------------------------------------------
    def apply_random_single(self, img: Image.Image, mask: Image.Image, n_ops=None):
        """Compose 2-4 single-image augmentations chosen at random."""
        ops = [
            # TODO: uncomment each entry once the corresponding method is implemented.
            ("rotate", self.rotate),
            # ("random_crop", self.random_crop),
            # ("flip", self.flip),
            # ("blur", self.blur),
            # ("scale", self.scale),
            # ("translate", self.translate),
            # ("partial_erase", self.partial_erase),
        ]
        if n_ops is None:  # the number of augmentations to apply
            n_ops = min(len(ops), int(self.rng.integers(2, 5)))

        idxs = self.rng.choice(len(ops), size=n_ops, replace=False)
        applied = []  # memo; which augmentations are applied
        for i in idxs.tolist():
            name, fn = ops[i]
            img, mask = fn(img, mask)
            applied.append(name)
        return img, mask, applied

    def apply_random_double(self, img: Image.Image, mask: Image.Image,
                            img_b: Image.Image, mask_b: Image.Image):
        """Pick one two-image augmentation at random and apply it once."""
        ops = [
            # TODO: uncomment each entry once the corresponding method is implemented.
            # ("stitch", self.stitch),
            # ("copy_paste", self.copy_paste),
        ]
        if not ops:  # TODO remove this part after implementing ops.
            return img, mask, ["no_aug"]

        i = int(self.rng.integers(0, len(ops)))
        name, fn = ops[i]
        out_img, out_mask = fn(img, mask, img_b, mask_b)
        return out_img, out_mask, [name]


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(root, "visualization")
    os.makedirs(out_dir, exist_ok=True)

    aug = SegAugmentation(seed=42)

    names = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(root, "img", "*.jpg")))

    # Probability of taking the two-image path (mutually exclusive with single).
    p_double = 0.3

    for name in names:
        img, mask = load_pair(root, name)
        if len(names) > 1 and aug.rng.random() < p_double:
            # On-the-fly pick and load a different sample as the secondary input.
            partner = name
            while partner == name:  # simple trick to select different image
                partner = names[int(aug.rng.integers(0, len(names)))]

            img_b, mask_b = load_pair(root, partner)
            a_img, a_mask, applied = aug.apply_random_double(img, mask, img_b, mask_b)
            memo = f"{applied[0]}(+{partner})"
        else:
            a_img, a_mask, applied = aug.apply_random_single(img, mask)
            memo = "+".join(applied)

        panel = make_panel(np.array(a_img), np.array(a_mask))
        out_path = os.path.join(out_dir, f"{name}.png")
        Image.fromarray(panel).save(out_path)
        print(f"{name}: {memo} -> {panel.shape[1]}x{panel.shape[0]}")


if __name__ == "__main__":
    main()
