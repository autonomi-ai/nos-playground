from dataclasses import dataclass

import torch
from transformers import pipeline

from nos.hub import HuggingFaceHubConfig

@dataclass(frozen=True)
class BarkConfig(HuggingFaceHubConfig):
    torch_dtype: str = "float16"
    """Torch dtype string to use for inference."""

class Bark:
    configs = {
        "suno/bark-small": BarkConfig(
            model_name="suno/bark-small",
            torch_dtype="float16",
        ),
        "suno/bark": BarkConfig(
            model_name="suno/bark",
            torch_dtype="float16",
        ),
    }

    def __init__(self, model_name: str = "suno/bark"):
        try:
            self.cfg = Bark.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {Bark.configs.keys()}")

        if torch.cuda.is_available():
            self.device_str = "cuda"
        else:
            self.device_str = "cpu"
        self.device = torch.device(self.device_str)

        self.pipe = pipeline("text-to-speech", self.cfg.model_name, device=self.device_str)
        self.pipe.model.to(self.device)
    
    def __call__(self, prompt: str):
        return self.pipe(prompt, forward_params={"do_sample": True})    
