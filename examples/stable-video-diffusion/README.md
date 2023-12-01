# Stable Video Diffusion
A nos integration example of [Stable Video Diffusion(SVD)](https://github.com/Stability-AI/generative-models). The SVD is a diffusion model trained for short video generation task, it uses a still image for conditioning to creates a short video clip. 
The orignal paper can be found [here](https://stability.ai/research/stable-video-diffusion-scaling-latent-video-diffusion-models-to-large-datasets).

## Deployment
Simply run the following command to serve up the model:
``` bash
nos serve up -c serve.yaml
```

## Inference
Here is an example how to run the inference:
``` python
import requests

from PIL import Image
from nos.client import Client

client = Client("[::]:50051")
model = client.Module("stable-video-diffusion")

image_link = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true"
image: Image.Image = requests.get(image_link)
response: Iterable[Image.Image] = model.image2video(image=image, _stream=True)
frames = list(response)

frames[0].save(
    "example.gif",
    save_all=True,
    append_images=frames[1:],
    optimize=False,
    duration=100,
    loop=0,
)
```
The "example.gif" should be looking like:
![output example](./example.gif)
