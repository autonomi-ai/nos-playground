from typing import Any, Dict, List, Union

import torch
from diffusers import DiffusionPipeline


class LatentConsistencyModel:
    def __init__(self, model_name: str = "SimianLuo/LCM_Dreamshaper_v7", dtype: str = "float32"):

        if torch.cuda.is_available():
            self.device_str = "cuda"
        else:
            self.device_str = "cpu"
        self.device = torch.device(self.device_str)
        self.torch_dtype = getattr(torch, dtype)

        self.pipe = DiffusionPipeline.from_pretrained(
            model_name, custom_pipeline="latent_consistency_txt2img", custom_revision="main"
        )
        self.pipe.to(torch_device=self.device_str, torch_dtype=self.torch_dtype)

    def __call__(
        self,
        prompts: Union[str, List[str]],
        negative_prompts: Union[str, List[str]] = None,
        num_images: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: int = -1,
    ) -> List[Dict[str, Any]]:
        """Generate images from text prompt."""
        with torch.inference_mode():
            torch.manual_seed(seed)
            result = self.pipe(
                prompt=prompts,
                negative_prompt=negative_prompts,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images,
                lcm_origin_steps=50,
                output_type="pil",
            ).images
        return result
