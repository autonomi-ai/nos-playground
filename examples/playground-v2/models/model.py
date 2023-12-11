from dataclasses import dataclass
from typing import List, Union

from PIL import Image
import torch
from diffusers import DiffusionPipeline

from nos.hub import HuggingFaceHubConfig

@dataclass(frozen=True)
class PlayGroundV2Config(HuggingFaceHubConfig):

    torch_dtype: str = "float16"
    """Torch dtype string to use for inference."""

    width: int = 1024
    """Image width to use for inference, optimally set to be the same as training."""

    height: int = 1024
    """Image height to use for inference, optimally set to be the same as training."""

class PlayGroundV2():

    configs = {
        "playgroundai/playground-v2-256px-base": PlayGroundV2Config(
            model_name="playgroundai/playground-v2-256px-base",
            torch_dtype="float16",
            width=256, height=256
        ),
        "playgroundai/playground-v2-512px-base": PlayGroundV2Config(
            model_name="playgroundai/playground-v2-512px-base",
            torch_dtype="float16",
            width=512, height=512
        ),
        "playgroundai/playground-v2-1024px-base": PlayGroundV2Config(
            model_name="playgroundai/playground-v2-1024px-base",
            torch_dtype="float16",
            width=1024, height=1024
        ),
        "playgroundai/playground-v2-1024px-aesthetic": PlayGroundV2Config(
            model_name="playgroundai/playground-v2-1024px-aesthetic",
            torch_dtype="float16",
            width=1024, height=1024
        ),
    }

    def __init__(self, model_name: str = "playgroundai/playground-v2-1024px-aesthetic"):

        # Assert that CUDA is available for GPU acceleration and onl use float16
        assert torch.cuda.is_available()

        self.pipe = DiffusionPipeline.from_pretrained(model_name, use_safetensors=True, torch_dtype=torch.float16, variant="fp16")
        self.pipe.to("cuda")

        self.pipe.enable_model_cpu_offload()

    def __call__(
        self,
        prompts: Union[str, List[str]],
        negative_prompts: Union[str, List[str]] = None,
        num_inference_steps: int = 30,
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

        # Generate images with the appropriate seed
        g = torch.Generator(device="cuda")
        if seed != -1:
            g.manual_seed(seed)
        else:
            g.seed()

        # The guidance_scale is set to 3.0 according to https://huggingface.co/playgroundai/playground-v2-1024px-aesthetic
        return self.pipe(
            prompts,
            negative_prompt=negative_prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=3.0,
            height=height,
            width=width,
            generator=g,
        ).images
