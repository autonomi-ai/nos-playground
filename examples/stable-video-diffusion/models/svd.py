from dataclasses import dataclass
from typing import Union, Iterable

from PIL import Image
import numpy as np
import torch

from nos.hub import HuggingFaceHubConfig

@dataclass(frozen=True)
class StableVideoDiffusionConfig(HuggingFaceHubConfig):
    
    model_cls: str = "StableVideoDiffusionPipeline"
    """Name of the model class to use."""

    torch_dtype: str = "float16"
    """Torch dtype string to use for inference."""

    width: int = 1024
    """Image width to use for inference, optimally set to be the same as training."""

    height: int = 576
    """Image height to use for inference, optimally set to be the same as training."""

    video_frames : int = 14
    """Number of frames the model is capable to generate."""

    decode_chunk_size : int = 8
    """Number of frames will be decoded at once."""

class StableVideoDiffusionModel():

    configs = {
        "stable-video-diffusion-img2vid": StableVideoDiffusionConfig(
            model_name="stabilityai/stable-video-diffusion-img2vid",
            video_frames=14
        ),
        "stable-video-diffusion-img2vid-xt": StableVideoDiffusionConfig(
            model_name="stabilityai/stable-video-diffusion-img2vid-xt",
            video_frames=25
        ),
    }
    
    def __init__(self, model_name: str = "stable-video-diffusion-img2vid-xt"):

        # Only support gpu for video generation
        assert torch.cuda.is_available()

        import diffusers

        try:
            self.cfg = StableVideoDiffusionModel.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {StableVideoDiffusionConfig.configs.keys()}")
        
        self.torch_dtype = getattr(torch, self.cfg.torch_dtype)
        self.variant = "fp16" if self.torch_dtype == torch.float16 else None

        model_cls = getattr(diffusers, self.cfg.model_cls)
        self.pipe = model_cls.from_pretrained(
            self.cfg.model_name,
            torch_dtype=self.torch_dtype,
            variant=self.variant
        )

        self.pipe.enable_model_cpu_offload()

    def image2video(
        self,
        image: Union[np.ndarray, Image.Image],
        seed: int = -1,
    ) -> Iterable[Image.Image]:
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.resize((self.cfg.width, self.cfg.height))

        generator = torch.manual_seed(seed)
        yield from self.pipe(image, decode_chunk_size=self.cfg.decode_chunk_size, generator=generator).frames[0]
