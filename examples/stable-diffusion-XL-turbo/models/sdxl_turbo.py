from typing import List, Union

from PIL import Image
import torch
from diffusers import AutoPipelineForText2Image

class StableDiffusionXLTurboModel:

    def __init__(self, model_name: str = "stabilityai/sdxl-turbo", dtype: str = "float16"):

        # Only support gpu for video generation
        assert torch.cuda.is_available()

        self.torch_dtype = getattr(torch, dtype)
        self.variant = "fp16" if self.torch_dtype == torch.float16 else None

        self.pipe = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=self.torch_dtype, variant="fp16")
        self.pipe.to("cuda")

    def __call__(
        self,
        prompts: Union[str, List[str]],
        negative_prompts: Union[str, List[str]] = None,
        num_images: int = 1,
        num_inference_steps: int = 1,
        height: int = None,
        width: int = None,
        seed: int = -1,
    ) -> List[Image.Image]:

        """Generate images from text prompt."""
        # Input validation and defaults
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
        if isinstance(negative_prompts, list):
            negative_prompts *= num_images

        # Generate images with the appropriate seed
        g = torch.Generator(device="cuda")
        if seed != -1:
            g.manual_seed(seed)
        else:
            g.seed()
        
        # The guidance_scale is set to 0.0 since the turbo model was trained without it.
        return self.pipe(
            prompts * num_images,
            negative_prompt=negative_prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
            height=height,
            width=width,
            generator=g,
        ).images
