import os
import glob
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

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
        w, h = img.size
        
        new_h = np.random.randint(int(h * size_frac[0]), int(h * size_frac[1]))
        new_w = np.random.randint(int(w * size_frac[0]), int(w * size_frac[1]))

        top  = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        cropped_img = img.crop((left, top, left + new_w, top + new_h))
        cropped_mask = mask.crop((left, top, left + new_w, top + new_h))
        
        return cropped_img, cropped_mask

    def flip(self, img: Image.Image, mask: Image.Image):
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        return flipped_img, flipped_mask

    def blur(self, img: Image.Image, mask: Image.Image, sigma_range=(0.8, 3.0)):
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        return blurred_img, mask

    def scale(self, img: Image.Image, mask: Image.Image, scale_range=(0.7, 1.3)):
        scale = np.random.uniform(scale_range[0], scale_range[1])
        w, h = img.size

        scale_size = (int(w * scale), int(h * scale))

        resized_img = img.resize(scale_size, resample=BILINEAR)
        resized_mask = mask.resize(scale_size, resample=NEAREST)

        return resized_img, resized_mask

    def translate(self, img: Image.Image, mask: Image.Image, max_frac: float = 0.2):
        w, h = img.size

        tx_frac = np.random.uniform(-max_frac, max_frac)
        ty_frac = np.random.uniform(-max_frac, max_frac)
        tx = tx_frac * w
        ty = ty_frac * h

        affine_matrix = (1, 0, -tx, 0, 1, -ty)

        translated_img = img.transform((w, h), Image.AFFINE, affine_matrix, resample=Image.BILINEAR)
        translated_mask = mask.transform((w, h), Image.AFFINE, affine_matrix, resample=Image.NEAREST)

        return translated_img, translated_mask

    def partial_erase(self, img: Image.Image, mask: Image.Image, frac_range=(0.1, 0.35)):
        w, h = img.size

        erase_w = int(w * np.random.uniform(*frac_range))
        erase_h = int(h * np.random.uniform(*frac_range))

        x = np.random.randint(0, w - erase_w + 1)
        y = np.random.randint(0, h - erase_h + 1)

        img_np = np.array(img)
        mean_pixel = np.mean(img_np, axis=(0, 1)).astype(int)
        fill_color = tuple(mean_pixel)

        res_img = img.copy()
        res_mask = mask.copy()
        draw_img = ImageDraw.Draw(res_img)
        draw_mask = ImageDraw.Draw(res_mask)

        draw_img.rectangle([x, y, x + erase_w, y + erase_h], fill=fill_color)
        draw_mask.rectangle([x, y, x + erase_w, y + erase_h], fill=0)

        return res_img, res_mask

    # --- Two-image augmentations (not included in apply_random) -------------

    def stitch(self, img_a: Image.Image, mask_a: Image.Image, img_b: Image.Image, mask_b: Image.Image):
        w_a, h_a = img_a.size
        w_b, h_b = img_b.size
        
        new_w_b = int(w_b * (h_a / h_b))

        img_b_resized = img_b.resize((new_w_b, h_a), resample=Image.BILINEAR)
        mask_b_resized = mask_b.resize((new_w_b, h_a), resample=Image.NEAREST)

        combined_img = Image.new(img_a.mode, (w_a + new_w_b, h_a))
        combined_mask = Image.new(mask_a.mode, (w_a + new_w_b, h_a))

        combined_img.paste(img_a, (0, 0))
        combined_img.paste(img_b_resized, (w_a, 0))
        combined_mask.paste(mask_a, (0, 0))
        combined_mask.paste(mask_b_resized, (w_a, 0))
        
        return combined_img, combined_mask

    def copy_paste(self, target_img: Image.Image, target_mask: Image.Image,
                source_img: Image.Image, source_mask: Image.Image):
        
        s_mask_np = np.array(source_mask)
    
        classes = np.unique(s_mask_np)
        classes = classes[(classes != 0) & (classes != 255)]
        
        if len(classes) == 0:
            return target_img, target_mask
        
        chosen_cls = np.random.choice(classes)
        
        coords = np.argwhere(s_mask_np == chosen_cls)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        
        obj_img = source_img.crop((x0, y0, x1, y1))
        obj_mask = source_mask.crop((x0, y0, x1, y1))
        
        tw, th = target_img.size
        ow, oh = obj_img.size
        
        if ow > tw or oh > th:
            scale = min(tw / ow, th / oh)
            new_size = (int(ow * scale), int(oh * scale))
            obj_img = obj_img.resize(new_size, Image.BILINEAR)
            obj_mask = obj_mask.resize(new_size, Image.NEAREST)
            ow, oh = new_size

        paste_x = np.random.randint(0, tw - ow + 1)
        paste_y = np.random.randint(0, th - oh + 1)
        
        res_img = target_img.copy()
        res_mask = target_mask.copy()
        
        obj_mask_np = np.array(obj_mask)
        binary_mask = Image.fromarray((obj_mask_np == chosen_cls).astype(np.uint8) * 255)
        
        res_img.paste(obj_img, (paste_x, paste_y), mask=binary_mask)
        res_mask.paste(obj_mask, (paste_x, paste_y), mask=binary_mask)
        
        return res_img, res_mask

    # --- Random composition --------------------------------------------------
    def apply_random_single(self, img: Image.Image, mask: Image.Image, n_ops=None):
        """Compose 2-4 single-image augmentations chosen at random."""
        ops = [
            # TODO: uncomment each entry once the corresponding method is implemented.
            ("rotate", self.rotate),
            ("random_crop", self.random_crop),
            ("flip", self.flip),
            ("blur", self.blur),
            ("scale", self.scale),
            ("translate", self.translate),
            ("partial_erase", self.partial_erase),
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
            ("stitch", self.stitch),
            ("copy_paste", self.copy_paste),
        ]

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
