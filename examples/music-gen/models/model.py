from dataclasses import dataclass

import torch
from transformers import pipeline

from nos.hub import HuggingFaceHubConfig

@dataclass(frozen=True)
class MusicGenConfig(HuggingFaceHubConfig):
    torch_dtype: str = "float16"
    """Torch dtype string to use for inference."""

class MusicGen:
    configs = {
        "facebook/musicgen-small": MusicGenConfig(
            model_name="facebook/musicgen-small",
            torch_dtype="float16",
        ),
        "facebook/musicgen-medium": MusicGenConfig(
            model_name="facebook/musicgen-medium",
            torch_dtype="float16",
        ),
        "facebook/musicgen-large": MusicGenConfig(
            model_name="facebook/musicgen-large",
            torch_dtype="float16",
        ),
    }

    def __init__(self, model_name: str = "facebook/musicgen-small"):
        try:
            self.cfg = MusicGen.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {MusicGen.configs.keys()}")

        if torch.cuda.is_available():
            self.device_str = "cuda"
        else:
            self.device_str = "cpu"
        self.device = torch.device(self.device_str)

        self.pipe = pipeline("text-to-audio", self.cfg.model_name, device=self.device_str)
        self.pipe.model.to(self.device)
    
    def __call__(self, prompt: str):
        return self.pipe(prompt, forward_params={"do_sample": True})    
