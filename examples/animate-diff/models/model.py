from typing import List, Union

from PIL import Image
import torch
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler

class AnimateDiff():

    def __init__(self, model_name: str = "Lykon/dreamshaper-7"):

        # Only support cuda and fp16 for video generation
        assert torch.cuda.is_available()

        adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", use_safetensors=True)
        self.pipe = AnimateDiffPipeline.from_pretrained(model_name, motion_adapter=adapter, use_safetensors=True)
        
        scheduler = DDIMScheduler.from_pretrained(
            model_name,
            subfolder="scheduler",
            beta_schedule="linear",
            clip_sample=False,
            timestep_spacing="linspace",
            steps_offset=1
        )

        self.pipe.scheduler = scheduler
        self.pipe.to(torch_device="cuda", torch_dtype=torch.float16)

        self.pipe.enable_vae_slicing()
        self.pipe.enable_model_cpu_offload()

    def __call__(
        self,
        prompts: Union[str, List[str]],
        negative_prompts: Union[str, List[str]] = None,
        num_inference_steps: int = 25,
        num_frames: int = 16,
        height: int = None,
        width: int = None,
        guidance_scale: float = 7.5,
        seed: int = -1,
    ) -> List[Image.Image]:
        """Generate animated images from text prompt."""

        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        g = torch.Generator(device="cuda")
        if seed != -1:
            g.manual_seed(seed)
        else:
            g.seed()

        yield from self.pipe(
            prompt=prompts,
            negative_prompt=negative_prompts,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=g,
            width=width,
            height=height,
        ).frames[0]
