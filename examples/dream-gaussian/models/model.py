import sys
from pathlib import Path

import numpy as np
import torch
from nos.common.git import cached_repo


def import_repo(*args, **kwargs) -> str:
    repo_dir = cached_repo(*args, **kwargs)
    sys.path.insert(0, repo_dir)
    return repo_dir


class DreamGaussian:
    def __init__(self):

        repo_path = Path(import_repo("https://github.com/jiexiong2016/dreamgaussian.git", force=True))
        from dream_gaussian import DreamGaussianModel

        assert torch.cuda.is_available(), "CUDA must be available to use DreamGaussian"

        self.device = torch.device("cuda")
        self.model = DreamGaussianModel(repo_path / Path("configs/dream_gsplat.yaml"))
        self.ref_size = 256

    def preprocess(self, img: np.ndarray):

        img = img.astype(np.float32) / 255.0

        input_alpha = img[..., 3:]
        input_img = img[..., :3] * input_alpha + (1 - input_alpha)
        input_img = input_img[..., ::-1].copy()

        input_img_torch = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        input_img_torch = torch.nn.functional.interpolate(
            input_img_torch, (self.ref_size, self.ref_size), mode="bilinear", align_corners=False
        )

        input_alpha_torch = torch.from_numpy(input_alpha).permute(2, 0, 1).unsqueeze(0).to(self.device)
        input_alpha_torch = torch.nn.functional.interpolate(
            input_alpha_torch, (self.ref_size, self.ref_size), mode="bilinear", align_corners=False
        )

        return input_img_torch, input_alpha_torch

    def __call__(self, img: np.ndarray):

        assert img.shape[2] == 4, "Input image must be in RGBA format"
        input_img_torch, input_alpha_torch = self.preprocess(img)

        self.model.train(input_img_torch, input_alpha_torch)
        return self.model.refine(input_img_torch)
