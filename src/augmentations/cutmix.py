import torch
import numpy as np


class CutMix:
    def __init__(self, alpha=1.0, probability=1.0, seed=None, rng=None):
        self.alpha = alpha
        self.probability = probability
        self.rng = rng or np.random.default_rng(seed)

    def rand_bbox(self, size, lam):
        """
        size = (B, C, H, W)
        """
        H = size[2]
        W = size[3]

        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = self.rng.integers(W)
        cy = self.rng.integers(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

    def __call__(self, images, labels):

        if self.rng.random() > self.probability:
            return images, labels, labels, 1.0

        batch_size = images.size(0)

        rand_index = torch.randperm(batch_size).to(images.device)

        labels_a = labels
        labels_b = labels[rand_index]

        lam = self.rng.beta(self.alpha, self.alpha)

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(images.size(), lam)

        images[:, :, bby1:bby2, bbx1:bbx2] = \
            images[rand_index, :, bby1:bby2, bbx1:bbx2]

        # Adjust lambda according to real patch area
        lam = 1 - (
            (bbx2 - bbx1) * (bby2 - bby1)
            / (images.size(-1) * images.size(-2))
        )

        return images, labels_a, labels_b, lam
