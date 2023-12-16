# Stable Diffusion XL Turbo
A nos integration example of [Stable Diffusion XL Turbo(SD-XL Turbo)](https://github.com/Stability-AI/generative-models).
The SD-XL Turbo is a diffusion model trained for real-time generation purpose, it is distilled from original SDXL 1.0 model.
The orignal paper can be found [here](https://stability.ai/research/adversarial-diffusion-distillation).

## Deployment
Simply run the following command to serve up the model:
``` bash
nos serve up -c serve.yaml
```

## Inference
Here is an example how to run the inference:
``` python
from PIL import Image
from nos.client import Client

client = Client("[::]:50051")
model = client.Module("sd-xl-turbo")

frames: List[Image.Image] = model(prompts=["astronaut on the moon, hdr, 4k"], num_inference_steps=1)
frames[0].save("example.png")
```
The "example.png" should be looking like:
![output example](./example.png)
