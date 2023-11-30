from typing import Any, Dict, List, Union

import numpy as np
import torch

from mvdream.camera_utils import get_camera
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.model_zoo import build_model

class MVDreamModel:
    def __init__(self, model_name: str = "sd-v2.1-base-4view", dtype: str = "float16"):

        if torch.cuda.is_available():
            self.device_str = "cuda"
        else:
            self.device_str = "cpu"
        self.torch_dtype = getattr(torch, dtype)

        self.model = build_model(model_name)
        self.model.device = self.device_str
        self.model.to(self.device_str)
        self.model.eval()

        self.sampler = DDIMSampler(self.model)

        # output 4 view with fixed angles
        self.nviews = 4
        self.camera = get_camera(self.nviews).to(self.device_str)

    def __call__(
        self,
        prompts: Union[str, List[str]],
        negative_prompts: Union[str, List[str]] = None,       
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        ddim_eta:float = 0.0,
        image_size: int = 256,
        seed: int = -1,
    ) -> List[Dict[str, Any]]:
        
        if isinstance(prompts, list):
            prompts = [prompts]

        if isinstance(negative_prompts, list):
            negative_prompts = [negative_prompts]
            
        if negative_prompts is None:
            negative_prompts = [""]

        with torch.no_grad(), torch.autocast(device_type=self.device_str, dtype=self.torch_dtype):
            torch.manual_seed(seed)
            
            c = self.model.get_learned_conditioning(prompts).to(self.device_str)
            uc = self.model.get_learned_conditioning(negative_prompts).to(self.device_str)

            c_ = {"context": c.repeat(self.nviews, 1, 1)}
            uc_ = {"context": uc.repeat(self.nviews,1, 1)}

            c_["camera"] = uc_["camera"] = self.camera
            c_["num_frames"] = uc_["num_frames"] = self.nviews

            shape = [4, image_size // 8, image_size // 8]
            samples_ddim, _ = self.sampler.sample(S=num_inference_steps, 
                                                conditioning=c_,
                                                batch_size=self.nviews, 
                                                shape=shape,
                                                verbose=False, 
                                                unconditional_guidance_scale=guidance_scale,
                                                unconditional_conditioning=uc_,
                                                eta=ddim_eta, 
                                                x_T=None)
            x_sample = self.model.decode_first_stage(samples_ddim)
            x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
            x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()
        return list(x_sample.astype(np.uint8))
