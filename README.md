# NOS Playground 🛝 
Playground for all kinds of examples with [NOS](https://github.com/autonomi-ai/nos).

<img src="assets/exp_txt2img.png" width="256">

## Installation
We assume the NOS server package and all its dependencies has been installed in prior. Check the [quick start document](https://docs.nos.run/docs/quickstart.html) from [NOS docs](https://docs.nos.run/) first if you haven't setup the develop enviroment yet.

## Quick Start
Simply run the following commands to serve the model you would like to deply: 
```console
cd examples/MODEL_ID
nos serve up -c serve.yaml
```
Then you can use the nos python client library to run the inference:
```python
from PIL import Image
from nos.client import Client

client = Client("[::]:50051")

model_id = "YOUR-MODEL-ID"
models: List[str] = client.ListModels()
assert model_id in models 
# Check if the selected model has been served.

inputs = YOUR-MODEL-INPUT
response = model(inputs) # Get output as response.
#change to model.DEFAULT_METHOD_NAME if the default method is defined as  "__call__"
```

## Available Examples

### Chat completion
```python
model_id: str = "meta-llama/Llama-2-7b-chat-hf"
```

### Video transcription
```python
model_id: str = "m-bain/whisperx-large-v2"
```

### Text to image
```python
model_id: List[str] = ["sd-xl-turbo", 
                        "playground-v2",
                        "latent-consistency-model"]
```
<img src="examples/stable-diffusion-XL-turbo/example.png" width="150">

### Text to video
```python
model_id: str = "animate-diff"
```
<img src="examples/animate-diff/example.gif" width="150">

### Image to video
```python
model_id: str = "stable-video-diffusion"
```
<img src="assets/exp_img2vid_in.png" width="150"><img src="assets/exp_img2vid_out.gif" width="150">

### Text to 360-view images
```python
model_id: str = "mv-dream"
```
<img src="examples/mvdream/example.png" width="600">

### Text to Speech
```python
model_id: str = "bark"
```
### Text to Music
```python
model_id: str = "music-gen"
```

## Reach US
* 💬 Send us an email at [support@autonomi.ai](mailto:support@autonomi.ai) or join our [Discord](https://discord.gg/QAGgvTuvgg) for help.
* 📣 Follow us on [Twitter](https://twitter.com/autonomi\_ai), and [LinkedIn](https://www.linkedin.com/company/autonomi-ai) to keep up-to-date on our products.
