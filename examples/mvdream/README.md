# MVDream
A nos integration example of [MVDream](https://github.com/bytedance/MVDream-threestudio), it is a multi-view diffusion model that is trained for generating multi-view images from given prompt. More techninal details can be found in [Paper](https://arxiv.org/abs/2308.16512)

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
model = client.Module("mv-dream")
response = model(prompts=["a rabbit sitting on the grass."])

img = np.concatenate(response, 1)
Image.fromarray(img).save(f"example.png")
```
This is how the "example.png" image should be looking like:
![output example](./example.png)
